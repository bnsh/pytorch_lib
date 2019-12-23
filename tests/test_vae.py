#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This program tests the VAE"""

import os
from PIL import Image, ImageDraw
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
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

#pylint: disable=too-many-locals
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data = MNISTish("/tmp/mnistvae/", train=True, download=True, transform=transform_mnist)
    data_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1)
    vae = VAE(1024, 2)
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
     LinearNorm(2, 1024),
     Reshape((-1, 1024, 1, 1)),
     nn.ConvTranspose2d(1024, 64, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0)),
     nn.RReLU(),
     nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
     nn.RReLU(),
     nn.ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
     nn.RReLU(),
     nn.ConvTranspose2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
     nn.RReLU(),
    ).to(device)

    for dummy_name, params in mnistvae.named_parameters():
        if len(params.shape) == 1:
            nn.init.zeros_(params)
        else:
            nn.init.orthogonal_(params)

    storefn = "/tmp/mnistvae/mnistvae.torch"
    if os.path.exists(storefn):
        mnistvae.load_state_dict(torch.load(storefn))

    opt = optim.Adamax(mnistvae.parameters())

    epoch = 0
    while True:
        total_vae = 0
        total_rec = 0
        denominator = 0

        mnistvae.eval()
        img = Image.new("RGBA", (2560, 2560))
        imgdraw = ImageDraw.ImageDraw(img)
        for idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)
            mnistvae(inputs)
            embed2d = vae.mean.detach().cpu().numpy()
            for label, pos in list(zip(targets, embed2d)):
                # Let's map these (arbitrarily) between -6 sd and 6 sd.
                # OR. Better yet, let's map them to the cdf's.
                xxx = norm.cdf(pos[0]) * 2560
                yyy = norm.cdf(pos[1]) * 2560
                # xxx = (pos[0] + 6) * 2560 / 12
                # yyy = (pos[1] + 6) * 2560 / 12
                imgdraw.ellipse((xxx-5, yyy-5, xxx+5, yyy+5), fill="hsl(%d,100%%,50%%)" % (label * 360 / 10))
        img.save("/tmp/mnistvae/%03d.png" % (epoch,))

        epoch += 1

        mnistvae.train(True)
        for idx, (inputs, dummy_targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            out = mnistvae(inputs)

            opt.zero_grad()
            reconstruction_loss = torch.mean(torch.pow(out - inputs, 2), dim=[1, 2, 3])
            vae_loss = vae.loss()
            vae_frac = 0.001
            batch_loss = (torch.sum(vae_frac * vae_loss + (1 - vae_frac) * reconstruction_loss))
            batch_loss.backward()

            total_vae = vae_loss.detach().sum().item()
            total_rec = reconstruction_loss.detach().sum().item()
            denominator += out.shape[0]
            opt.step()
            print("%d(%d/%d): vae=%.7f rec=%.7f" % (epoch, (1+idx), len(data_loader), total_vae/denominator, total_rec/denominator))


        torch.save(mnistvae.state_dict(), storefn)
#pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
