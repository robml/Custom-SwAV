import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import get_model

import apex
from apex.parallel.LARC import LARC

# helper functions #
def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# ################ #

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
args = parser.parse_args()

fixed_random_seeds() # set fixed random state
EPOCHS = 100

# Create train dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = CustomDataset(root='/dataset', split="train", transform=train_transform)
sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=2, sampler=sampler, pin_memory=True, drop_last=True)

# Create Model
net = get_model()
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = net.cuda()

# Optimizer
#optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
optimizer = torch.optim.SGD(
        model.parameters(),
        lr=4.8,
        momentum=0.9,
        weight_decay=1e-6,
    )
optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
warmup_lr_schedule = np.linspace(0, 4.8, len(train_loader) * 10)
iters = np.arange(len(train_loader) * (EPOCHS - 10))
cosine_lr_schedule = np.array([0 + 0.5 * (4.8 - 0) * (1 + \
                        math.cos(math.pi * t / (len(train_loader) * (EPOCHS - 10)))) for t in iters])
lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")

# Data Parallel Model
net = torch.nn.DataParallel(net)

# Criterion
#criterion = nn.CrossEntropyLoss()

# misc
cudnn.benchmark = True

# train function #
def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    losses = AverageMeter()
    running_loss = 0.0

    net.train()
    use_the_queue = False

    for it, data in enumerate(train_loader):
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = net.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        #outputs = net(inputs)
        #loss = criterion(outputs, labels)

        embedding, output = net(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate([0,1]):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            net.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum([2])), crop_id):
                x = output[bs * v: bs * (v + 1)] / 0.1
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # cancel gradients for the prototypes
        if iteration < 313:
            for name, p in net.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        
    return (epoch, losses.avg), queue

# SwAV specific function
@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / 0.05).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

print('Start Training')

net.train()
for epoch in range(EPOCHS):
    train_loader.sampler.set_epoch(epoch)
    scores, queue = train(train_loader,net,optimizer,epoch,lr_schedule,queue)
        
print('Finished Training')

os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(net.module.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))

print(f"Saved checkpoint to {os.path.join(args.checkpoint_dir, 'model.pth')}")