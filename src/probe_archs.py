import torch
import torch.nn.functional as F
from torch import nn

from .probe_training import Probe


class LinearProbe(Probe):
    # simple linear classifier for transformer activations
    
    def __init__(self, d_model):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class NonlinearProbe(Probe):
    # two-layer classifier with relu and dropout for transformer activations
    
    def __init__(self, d_model, d_mlp, dropout=0.1):
        super(NonlinearProbe, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)