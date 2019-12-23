#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Identity simply copies it's inputs. WHY?! Well. Sometimes it's useful to just
   replace a layer with a "NOOP", and this basically just does that."""

import torch.nn as nn

class Identity(nn.Module):
    #pylint: disable=arguments-differ
    def forward(self, inp):
        return 1.0 * inp
    #pylint: disable=arguments-differ
