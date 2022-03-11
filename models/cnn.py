from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, entity_dim):
        super(CNN, self).__init__()
        self.dim = entity_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10,
                               kernel_size=6)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10,
                               kernel_size=6)
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.fc1 = nn.Linear(int(self.dim / 25 - 2) * 10, self.dim)
        self.fc2 = nn.Linear(self.dim, self.dim * 2)
        self.fc3 = nn.Linear(self.dim * 2, self.dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):

    def __init__(self, entity_dim):
        super(CNN2, self).__init__()
        self.dim = entity_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10,
                               kernel_size=6)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10,
                               kernel_size=6)
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.fc1 = nn.Linear(2 * int(self.dim / 25 - 2) * 10, 2 * self.dim)
        self.fc3 = nn.Linear(self.dim * 2, self.dim)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.pool(self.conv1(x1))
        x1 = self.pool(self.conv2(x1))

        x2 = x2.unsqueeze(1)
        x2 = self.pool(self.conv1(x2))
        x2 = self.pool(self.conv2(x2))

        x = torch.cat((x1, x2), dim=-1)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return x