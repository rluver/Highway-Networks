import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(HighwayNetwork, self).__init__()
        self.H = nn.Linear(in_features=kwargs.get('in_features'), out_features=kwargs.get('in_features'), bias=-1)
        self.T = nn.Linear(in_features=kwargs.get('in_features'), out_features=kwargs.get('in_features'))

    def forward(self, x):
        H = self.H(x)
        H = F.relu(H)
        T = self.T(x)
        T = F.sigmoid(T)

        output = H*T + x * (1-T)

        return output
