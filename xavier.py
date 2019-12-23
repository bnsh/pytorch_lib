#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""Xavier does Xavier initialization given a _module_"""

import torch.nn.init as init

def xavier(params):
    for param in params.parameters():
        if param.detach().ndimension() == 2:
            init.xavier_uniform(param.detach())
        else:
            param.detach().fill_(1.0)
    return params
