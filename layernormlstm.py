#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""
This was taken directly from https://github.com/sosuperic/sketching-with-language/blob/master/src/models/core/layernormlstm.py
    (I made it pylint clean is all)
    Also, I added the "batch_first" argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    # Unfortunately, nn.Module has args as "*args" but
    # nn.LSTMCell has args as inps, hx=None, so it's impossible to
    # avoid this "arguments-differ.
    #pylint: disable=arguments-differ,invalid-name
    def forward(self, inps, hidden=None):
        self.check_forward_input(inps)
        if hidden is None:
            hx = inps.new_zeros(inps.size(0), self.hidden_size, requires_grad=False)
            cx = inps.new_zeros(inps.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(inps, hx, '[0]')
        self.check_forward_hidden(inps, cx, '[1]')

        gates = self.ln_ih(F.linear(inps, self.weight_ih, self.bias_ih)) \
                 + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * self.dropout(g))
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy
    #pylint: enable=arguments-differ,invalid-name


class LayerNormLSTM(nn.Module):

    #pylint: disable=too-many-arguments
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, bidirectional=False, batch_first=False, rec_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
         LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions), hidden_size=hidden_size, bias=bias, dropout=rec_dropout) for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
             LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions), hidden_size=hidden_size, bias=bias, dropout=rec_dropout) for layer in range(num_layers)
            ])
    #pylint: enable=too-many-arguments

    # Unfortunately, nn.Module has args as "*args" but
    # nn.LSTMCell has args as inps, hx=None, so it's impossible to
    # avoid this "arguments-differ.
    #pylint: disable=arguments-differ,too-many-locals,invalid-name
    def forward(self, inps, hidden=None):
        if self.batch_first:
            inps = inps.transpose(1, 0, 2)
        seq_len, batch_size, dummy_hidden_size = inps.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = inps.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = inps.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = inps
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(inps):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        if self.batch_first:
            y = y.transpose(1, 0, 2)
        return y, (hy, cy)
    #pylint: enable=arguments-differ,too-many-locals,invalid-name
