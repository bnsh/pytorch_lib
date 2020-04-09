#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This program tests the VAE"""

import os
from PIL import Image, ImageDraw
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST as MNISTish
from tqdm import tqdm
from pytorchlib.vae import VAE
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

class Parallel(nn.Module):
    def __init__(self, modules):
        super(Parallel, self).__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, *raw):
        inps, *_ = raw

        retval = [mod(inps) for mod in self.module_list]
        children = [submodule for module in self.module_list[1].children() for submodule in module.children()]
        out = inps
        out = children[0](out)
        out = children[1](out)
        out = children[2](out)
        out = children[3](out)
        return retval


def scale_func(epoch):
    """
        Want a function such that
        f(0) = 0.001
        f(100) = 0.3
        f(Infinity) = 1/3
        Make it be a sigmoid?

        a f(0) + b = low
        a f(mx k) + b = mid
        a f(Infinity) + b = high
        #
        a/2 + b = low
        a + b = high
        a/2 = (high-low)
        a = 2 * (high-low)
        (high-low) + b = low
        b = low - high + low
        b = 2 * low - high
        #
        2 * (high-low) f(mx k) + 2 * low - high = mid
        2 * (high-low) f(mx k) = mid - 2 * low + high
        f(mx k) = (mid - 2 * low + high) / (2 * (high-low))
        (1 + Exp[-mx k])^-1 = (mid - 2 * low + high) / (2 * (high-low))
        (1 + Exp[-mx k]) = (2 * (high-low)) / (mid - 2 * low + high)
        (Exp[-mx k]) = (2 * (high-low) - mid + 2 * low - high) / (mid - 2 * low + high)
        (Exp[-mx k]) = (2 * high - 2 * low - mid + 2 * low - high) / (mid - 2 * low + high)
        (Exp[-mx k]) = (high - mid) / (mid - 2 * low + high)
        -mx k = Log[(high - mid)] - Log[(mid - 2 * low + high)]
        mx k = Log[(mid - 2 * low + high)] - Log[(high - mid)]
        k = Log[(mid - 2 * low + high)] - Log[(high - mid)] / mx
    """
    low = 0.00
    high = 1.0/3.0
    mxx = 100
    mid = 1/6 # f(mxx) = mid
    aaa = 2 * (high-low)
    bbb = 2 * low - high
    kkk = (np.log(mid - 2 * low + high) - np.log(high - mid)) / mxx

    return bbb + aaa / (1.0 + np.exp(-epoch * kkk))

def create_network(device, embeddingsz):
    vae = VAE(1024, embeddingsz)
    mnistvae = nn.Sequential(
     nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.RReLU(),
     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28x28x32
     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 14x14x32
     nn.RReLU(),
     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 14x14x32
     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 7x7x64
     nn.RReLU(),
     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 7x7x64
     nn.RReLU(),
     nn.Conv2d(64, 1024, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0)), # 1x1x1024
     nn.Flatten(), # 1024
     vae,
     Parallel([
      nn.Sequential(
       LinearNorm(embeddingsz, 1024),
       Reshape((-1, 1024, 1, 1)),
       nn.ConvTranspose2d(1024, 64, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0)),
       nn.RReLU(),
       nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
       nn.RReLU(),
       nn.ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
       nn.RReLU(),
       nn.ConvTranspose2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
       nn.RReLU(),
      ),
      nn.Sequential(
       LinearNorm(embeddingsz, 8),
       LinearNorm(8, 10)
      )
     ])
    ).to(device)

    return vae, mnistvae

#pylint: disable=too-many-arguments,too-many-locals
def evaluate(device, dirname, epoch, vae, mnistvae, data_loader_eval, calibration):
    calib_vae, calib_rec, calib_cls = calibration
    total_vae = 0
    total_rec = 0
    total_cls = 0
    correct = 0
    denominator = 0

    mnistvae.eval()
    img = Image.new("RGBA", (5120, 2560))
    imgdraw = ImageDraw.ImageDraw(img)
    for _, (inputs, targets) in enumerate(tqdm(data_loader_eval, dynamic_ncols=True, leave=False, desc="image")):
        inputs = inputs.to(device)
        targets = targets.type(torch.long).to(device)
        reconstructed, classification = mnistvae(inputs)

        reconstruction_loss = torch.sum(torch.pow(reconstructed - inputs, 2))
        classification_loss = F.cross_entropy(classification, targets, reduction="sum")
        vae_loss = vae.loss().sum()

        total_vae += vae_loss.detach().sum().item()
        total_rec += reconstruction_loss.detach().sum().item()
        total_cls += classification_loss.detach().sum().item()
        correct += (torch.argmax(classification, dim=1) == targets.detach()).sum().cpu().item()
        denominator += reconstructed.shape[0]

        embed2d = vae.mean.detach().cpu().numpy()
        for label, pos in list(zip(targets, embed2d)):
            xxx = np.clip((pos[0] + 6) * 2560 / 12, 0, 2560)
            yyy = np.clip((pos[1] + 6) * 2560 / 12, 0, 2560)
            imgdraw.ellipse((xxx-5, yyy-5, xxx+5, yyy+5), fill="hsl(%d,100%%,50%%)" % (label * 360 / 10))
            # Let's map these (arbitrarily) between -6 sd and 6 sd.
            # OR. Better yet, let's map them to the cdf's.
            xxx = 2560 + np.clip(norm.cdf(pos[0]) * 2560, 0, 2560)
            yyy = np.clip(norm.cdf(pos[1]) * 2560, 0, 2560)
            imgdraw.ellipse((xxx-5, yyy-5, xxx+5, yyy+5), fill="hsl(%d,100%%,50%%)" % (label * 360 / 10))
    img.save(os.path.join(dirname, "%03d.png" % (epoch,)))
    correct = correct / denominator

    total_vae = total_vae / denominator
    if calib_vae is None:
        calib_vae = total_vae
    calib_vae = 0.9 * calib_vae + 0.1 * total_vae

    total_rec = total_rec / denominator
    if calib_rec is None:
        calib_rec = total_rec
    calib_rec = 0.9 * calib_rec + 0.1 * total_rec

    total_cls = total_cls / denominator
    if calib_cls is None:
        calib_cls = total_cls
    calib_cls = 0.9 * calib_cls + 0.1 * total_cls

    print("%d: vae=%.7f rec=%.7f class=%.7f accuracy=%.7f" % (epoch, total_vae, total_rec, total_cls, correct))
    calibration[0] = calib_vae
    calibration[1] = calib_rec
    calibration[2] = calib_cls
#pylint: enable=too-many-arguments,too-many-locals

#pylint: disable=too-many-arguments,too-many-locals
def train(device, epoch, opt, vae, mnistvae, data_loader_train, calibration):
    calib_vae, calib_rec, calib_cls = calibration
    vae_frac = scale_func(epoch)

    mnistvae.train(True)
    for _, (inputs, targets) in enumerate(tqdm(data_loader_train, dynamic_ncols=True, leave=False, desc="learn")):
        inputs = inputs.to(device)
        targets = targets.type(torch.long).to(device)
        reconstructed, classification = mnistvae(inputs)

        opt.zero_grad()
        reconstruction_loss = torch.sum(torch.pow(reconstructed - inputs, 2)) / reconstructed.shape[0]
        classification_loss = F.cross_entropy(classification, targets, reduction="sum") / reconstructed.shape[0]
        vae_loss = vae.loss().sum() / reconstructed.shape[0]

        # classification and reconstruction should be equally good.
        batch_loss = torch.sum(vae_frac * (vae_loss / calib_vae) + ((1-vae_frac)/2.0) * (reconstruction_loss / calib_rec) + ((1-vae_frac)/2.0) * (classification_loss / calib_cls))
        batch_loss.backward()

        opt.step()
#pylint: enable=too-many-arguments,too-many-locals

def main():
    dirname = "/tmp/mnistvae_supervised"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data = MNISTish(dirname, train=True, download=True, transform=transform_mnist)
    validation_data = MNISTish(dirname, train=False, download=True, transform=transform_mnist)
    data_loader_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    data_loader_eval = DataLoader(validation_data, batch_size=256, shuffle=True, num_workers=1)

    vae, mnistvae = create_network(device, 2)

    for name, params in mnistvae.named_parameters():
        if len(params.shape) == 1:
            if name.endswith("weight"):
                nn.init.ones_(params)
            else:
                nn.init.zeros_(params)
        else:
            nn.init.orthogonal_(params)

    storefn = os.path.join(dirname, "mnistvae_supervised.torch")
    if os.path.exists(storefn):
        mnistvae.load_state_dict(torch.load(storefn), strict=False)

    opt = optim.Adamax(mnistvae.parameters())

    epoch = 0
    calibration = [None, None, None]

    for eee in (0, 10, 100, 1000, 10000):
        print(eee, scale_func(eee))
    while True:
        evaluate(device, dirname, epoch, vae, mnistvae, data_loader_eval, calibration)
        epoch += 1
        train(device, epoch, opt, vae, mnistvae, data_loader_train, calibration)
        torch.save(mnistvae.state_dict(), storefn)

if __name__ == "__main__":
    main()
