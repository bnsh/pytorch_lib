#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""This will contain any stats stuff that I don't find in torch proper."""

import torch
from torch.autograd import Variable

def correlation_coefficient(data_a, data_b):
    # first, compute the mean and standard deviation of each?
    epsilon = Variable(torch.FloatTensor([1e-6]).type_as(data_a.detach()))
    mean_a = torch.mean(data_a, dim=0, keepdim=True)
    mean_b = torch.mean(data_b, dim=0, keepdim=True)

    # I use unbiased=False to make it match scipy.stats.stats.pearsonr
    # (also np.std)
    std_a = torch.max(torch.std(data_a, unbiased=False, dim=0, keepdim=True), epsilon)
    std_b = torch.max(torch.std(data_b, unbiased=False, dim=0, keepdim=True), epsilon)

    adjusted_a = (data_a - mean_a)
    adjusted_b = (data_b - mean_b)
    numerator = torch.matmul(adjusted_a.t(), adjusted_b) / data_a.shape[0]
    denominator = torch.matmul(std_a.t(), std_b)
    return numerator / denominator
