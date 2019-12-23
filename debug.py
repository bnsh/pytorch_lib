#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""
    Debug will take a module and run the module, but store it's input and output
    and gradients and gradients in a directory
"""

import os
import torch
import torch.nn as nn

class Debug(nn.Module):
    def __init__(self, module, directory, label, only_debug_while_training=True):
        super(Debug, self).__init__()
        self.directory = os.path.join(directory, label)
        self.index = 0
        self.label = label
        self.only_debug_while_training = only_debug_while_training
        self.add_module(self.label, module)
        self.remove_handle = self.register_backward_hook(self.backward_hook)

    @staticmethod
    def save(data, filename):
        assert not os.path.exists(filename)
        torch.save(data, filename)

    def forward(self, *rawdata):
        self.index += 1
        inp, = rawdata

        module = getattr(self, self.label)
        out = module(inp)

        if (self.only_debug_while_training and self.training) or (not self.only_debug_while_training):
            if os.path.exists(self.directory):
                self.save(inp.cpu(), os.path.join(self.directory, "%08d-input.torch" % (self.index)))
                self.save(out.cpu(), os.path.join(self.directory, "%08d-output.torch" % (self.index)))

        return out

    def backward_hook(self, _, dedinput, dedoutput):
        if os.path.exists(self.directory):
            self.save(dedinput, os.path.join(self.directory, "%08d-dEdinput.torch" % (self.index)))
            self.save(dedoutput, os.path.join(self.directory, "%08d-dEdoutput.torch" % (self.index)))
