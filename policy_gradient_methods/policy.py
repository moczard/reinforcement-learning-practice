from torch import nn

import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        hidden_dim = 128
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(0.6)
        self.lin2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.softmax(x, dim=1)
