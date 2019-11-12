# Policy and value function (network) models

import torch, torch.nn as nn
import torch.nn.functional as F

class TwoLayerFCNet(nn.Module):
    def __init__(self, n_in=4, n_hidden=128, n_out=2):
        super().__init__()
        ### <<< Your Code Here
        self.fc1 = nn.Linear('???', '???')
        self.fc2 = '???'
        ### Your Code Ends >>>

    def forward(self, x):
        ### <<< Your Code Here
        x = F.relu('???')
        ### Your Code Ends >>>
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, n_in=[4,84,84], conv_channels=[32, 64, 64],
                 conv_kernels=[8, 4, 3], conv_strides=[4, 2, 1], n_fc=[256], n_out=4):
        super().__init__()

        self.conv_layers = []
        c0 = n_in[0]
        h0 = n_in[1]
        assert n_in[1] == n_in[2], 'input must be square image'
        for c, k, s in zip(conv_channels, conv_kernels, conv_strides):
            
            ### <<< Your Code Here
            # append nn.Conv2d with kernel size k, stride s
            self.conv_layers.append( '???' )
            # append nn.ReLU layer
            self.conv_layers.append( '???' )
            ### Your Code Ends >>>

            h0 = int(float(h0-k) / s + 1)
            c0 = c
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.fc_layers = []
        h0 = h0 * h0 * conv_channels[-1]
        for i, h in enumerate(n_fc):
            # append Linear and ReLU layers
            ### <<< Your Code Here:
            self.fc_layers.append( '???' )
            ### Your Code Ends >>>
            
            self.fc_layers.append( nn.ReLU() )
            h0 = h
        if type(n_out) is list:
            self.out_layers = nn.ModuleList([ nn.Linear(h, o) for o in n_out ])
        else:
            self.fc_layers.append( nn.Linear(h, n_out) )
        self.fc_layers = nn.Sequential(*self.fc_layers)


    def forward(self, x, head=None):
        x = x.float() / 256

        ### <<< Your Code Here:
        # feed x into the self.conv_layers
        x = '???'
        # (flatten) reshape x into a batch of vectors
        x = '???'
        # feed x into the self.fc_layers
        x = '???'
        ### Your Code Ends >>>

        if head is not None:
            x = self.out_layers[head](x)
        return x

class DuelQNet(nn.Module):
    def __init__(self):
        pass