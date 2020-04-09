#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Let's see. Maybe we'll test the "attention" mechanism with
   MNIST."""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST as MNISTish

from pytorchlib.attention import Attention
from pytorchlib.linearnorm import LinearNorm

def transform_mnist(pil):
    retval = np.array(pil, dtype=np.float32).reshape((28, 28, 1)).transpose(2, 0, 1)
    retval = (retval / 255.0) * 2.0 - 1.0
    return retval

# *Sigh* No reshape module in pytorch
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, *raw):
        inps, *_ = raw
        return torch.reshape(inps, self.shape)

def create_network(device):
    retval = nn.Sequential(
     nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.RReLU(),
     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.RReLU(),
     nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.RReLU(),
     nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.RReLU(),
     Reshape((-1, 784, 256)),
     Attention(256, 1),
     LinearNorm(256, 256),
     nn.RReLU(),
     LinearNorm(256, 10),
    ).to(device)
    return retval

def dump_batch(fname, squaresz, data):
    img = Image.new("RGBA", (squaresz*28, squaresz*28))
    for idx1 in range(0, squaresz):
        for idx2 in range(0, squaresz):
            idx = idx1 * squaresz + idx2
            dummy_loss, dummy_predicted, dummy_target, subimg = data[idx]

            subimg = Image.fromarray(np.array(subimg, dtype=np.uint8))
            img.paste(subimg, box=(idx1*28, idx2*28))

    img.save(fname)

#pylint: disable=too-many-locals
def evaluate(device, dirname, epoch, net, data_loader_eval):
    total_cls = 0
    correct = 0
    denominator = 0
    net.eval()

    pix = []
    for _, (inputs, targets) in enumerate(tqdm(data_loader_eval, ncols=0, leave=False, desc="eval")):
        inputs = inputs.to(device)
        targets = targets.type(torch.long).to(device)

        classification = net(inputs)

        indloss = F.cross_entropy(classification, targets, reduction="none")
        classification_loss = F.cross_entropy(classification, targets, reduction="sum")
        total_cls += classification_loss.detach().sum().item()
        predicted = torch.argmax(classification.detach(), dim=-1)
        correct += (predicted == targets.detach()).sum().cpu().item()
        denominator += classification.shape[0]

        # Now, let's construct images that are colored by attention
        attention = list(net.children())[9].focus.detach().reshape(-1, 1, 28, 28).permute(0, 2, 3, 1)
        logattention = torch.log(attention)
        meanval = logattention.mean(dim=[1, 2, 3], keepdim=True)
        stdval = logattention.std(dim=[1, 2, 3], keepdim=True)
        z_attention = (logattention - meanval) / stdval
        scaled_attention = torch.sigmoid(100 * z_attention)
# We want to map
# -a + b == sw
#  a + b == 1-sw
# 2b = 1.0
# b = 1/2
# 2a = 1-2sw
# a = 0.5 - sw
# f(x) = (1/2-sw) x + 1/2
        pixels = (((0.5-0.25) * inputs + 0.5)).permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        red = scaled_attention*2
        grn = torch.ones(scaled_attention.shape).type_as(scaled_attention)
        blu = 1-scaled_attention*2
        combined = torch.clamp(torch.cat([red, grn, blu], dim=-1) * pixels * 255, 0, 255)

        pix.extend(list(zip(indloss.detach().cpu().numpy().tolist(), predicted.detach().cpu().numpy().tolist(), targets.detach().cpu().numpy().tolist(), combined.detach().cpu().numpy().tolist())))

    total_cls = total_cls / denominator
    correct = correct / denominator

    squaresz = 10
    # Now, take the top squaresz*squaresz values (and the bottom) and draw images.
    pix = sorted(pix, key=lambda x: x[0])
    best = pix[0:(squaresz*squaresz)]
    worst = pix[-(squaresz*squaresz):]
    dump_batch(os.path.join(dirname, "worst-%d.png" % (epoch,)), squaresz, worst)
    dump_batch(os.path.join(dirname, "best-%d.png" % (epoch,)), squaresz, best)

    print("%d: loss=%.7f accuracy=%.7f" % (epoch, total_cls, correct))
#pylint: disable=too-many-locals

def train(device, opt, net, data_loader_train):
    net.train(True)
    for _, (inputs, targets) in enumerate(tqdm(data_loader_train, ncols=0, leave=False, desc="learn")):
        inputs = inputs.to(device)
        targets = targets.type(torch.long).to(device)
        classification = net(inputs)
        opt.zero_grad()

        classification_loss = F.cross_entropy(classification, targets, reduction="sum") / inputs.shape[0]
        classification_loss.backward()
        opt.step()

def main():
    dirname = "/tmp/mnistattention"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data = MNISTish(dirname, train=True, download=True, transform=transform_mnist)
    validation_data = MNISTish(dirname, train=False, download=True, transform=transform_mnist)
    data_loader_train = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    data_loader_eval = DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=1)

    net = create_network(device)

    storefn = os.path.join(dirname, "mnistattention.torch")
    if os.path.exists(storefn):
        net.load_state_dict(torch.load(storefn), strict=False)
    opt = optim.Adamax(net.parameters())

    epoch = 0
    while True:
        evaluate(device, dirname, epoch, net, data_loader_eval)
        epoch += 1
        train(device, opt, net, data_loader_train)
        torch.save(net.state_dict(), storefn)

if __name__ == "__main__":
    main()
