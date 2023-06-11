import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--log-dir",default='log',type=str)
parser.add_argument("--exp-name",default=None,type=str)
parser.add_argument("--low-fusion",default=None,type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

def get_logger(dataset_name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = f"{dataset_name}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger

def get_writer(args, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        f.writelines([f"{k}: {v}\n" for k, v in vars(args).items()])
    writer = SummaryWriter(log_dir)
    return writer

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# logger
args.exp_name = args.exp_name if args.exp_name else datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
exp_dir = os.path.join(args.log_dir, args.exp_name)
logger = get_logger("animaltrack", exp_dir)
# writer
writer = get_writer(args, exp_dir)

# data loading
# root = args.data_dir
# train_dir = os.path.join(root,"train")
# test_dir = os.path.join(root,"test")
# transform_train = torchvision.transforms.Compose([
#     # torchvision.transforms.RandomCrop((128,64),padding=4),
#     torchvision.transforms.Resize((128,64)),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# transform_test = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((128,64)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# trainloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
#     batch_size=64,shuffle=True
# )
# testloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
#     batch_size=64,shuffle=True
# )
# num_classes = len(trainloader.dataset.classes)
data_dir = args.data_dir
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
all_data = torchvision.datasets.ImageFolder(data_dir, transform=transform)
test_size = 0.2
indices = list(range(len(all_data)))
np.random.shuffle(indices)
split_idx = int(np.floor(test_size * len(all_data)))
train_idx, test_idx = indices[split_idx:], indices[:split_idx]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

trainloader = torch.utils.data.DataLoader(all_data, batch_size=64, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(all_data, batch_size=64, sampler=test_sampler)
assert len(trainloader.dataset.classes) == len(testloader.dataset.classes)
num_classes = len(trainloader.dataset.classes)

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile(f"./checkpoint/{args.exp_name}.t7"), "Error: no checkpoint file found!"
    logger.info(f'Loading from checkpoint/{args.exp_name}.t7')
    checkpoint = torch.load(f"./checkpoint/{args.exp_name}.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# train function for each epoch
def train(epoch):
    logger.info("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx+1)%interval == 0:
            end = time.time()
            logger.info("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        logger.info("Testing ...")
        end = time.time()
        logger.info("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        logger.info(f"Saving parameters to checkpoint/{args.exp_name}.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, f'./checkpoint/{args.exp_name}.t7')

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        logger.info("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch, start_epoch+10):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Error/train", train_err, epoch)
        writer.add_scalar("Error/test", test_err, epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
