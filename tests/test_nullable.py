#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Test the nullable module"""

import torch
import torch.nn as nn
from pytorchlib.nullable import Nullable

def zerofill(_, shape):
    return torch.zeros(shape)

def test(label, mod, ind, data):
    inp = {"data": data, "indicator": ind}
    print(label)
    print(mod(inp))

def main():
    mod = Nullable(nn.Sequential(
     nn.Linear(2, 5),
     nn.Tanh()
    ), zerofill)

    ind0 = torch.zeros(10, 1)
    ind1 = torch.ones(10, 1)
    indb = torch.zeros(10, 1).bernoulli(0.5)
    data = torch.randn(10, 2)

    test("indicator is all zero", mod, ind0, data)
    test("indicator is all ones", mod, ind1, data)
    test("indicator is bernoulli", mod, indb, data)


if __name__ == "__main__":
    main()
