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
from PIL import Image
import pandas as pd

from model import TripleNet

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

class TripleDataset:
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.df.iloc[idx, 0]))
        pos_image = self.transform(Image.open(self.df.iloc[idx, 1]))
        neg_image = self.transform(Image.open(self.df.iloc[idx, 2]))
        return (image, pos_image, neg_image)


# train function for each epoch
def train(epoch, loader, net):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=1e-3)
    logger.info("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    interval = args.interval
    start = time.time()

    for idx, (img, pos, neg) in enumerate(loader):
        # forward
        img, pos, neg = img.to(device), pos.to(device), neg.to(device)
        out = net(img)
        pos_out = net(pos)
        neg_out = net(neg)
        pos_sim = out @ pos_out.T
        neg_sim = out @ neg_out.T
        pos_lab = torch.ones(pos_sim.shape).to(device)
        neg_lab = torch.zeros(neg_sim.shape).to(device)
        outputs = torch.cat([pos_sim, neg_sim], axis=1)
        labels = torch.cat([pos_lab, neg_lab], axis=1)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()

        # print
        if (idx+1)%interval == 0:
            end = time.time()
            logger.info("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f}".format(
                100.*(idx+1)/len(loader), end-start, training_loss/interval,
            ))
            training_loss = 0.
            start = time.time()
    
    return net, train_loss/len(loader)


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    df = pd.read_csv("/home/kwangeunyeo/MAT/deep_sort/deep/triplet_dataset.csv")
    dataset = TripleDataset(df, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    net = TripleNet()
    net = net.to(device)

    # loss and optimizer
    for epoch in range(5):
        net, train_loss = train(epoch, dataloader, net)
        writer.add_scalar("Loss/train", train_loss, epoch)

        logger.info(f"Saving parameters to checkpoint/{args.exp_name}.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':0.,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, f'./checkpoint/{args.exp_name}_{epoch}.t7')

    writer.flush()
    writer.close()



if __name__ == '__main__':
    main()
