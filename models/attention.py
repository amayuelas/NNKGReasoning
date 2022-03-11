import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AndAttention(nn.Module):
    def __init__(self, n_layers, entity_dim, temperature, attn_dropout=0.1):
        super(AndAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_layers = n_layers
        for i in range(1, self.n_layers + 1):
            setattr(self, "and_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)


    def forward(self, x1, x2):
        x = torch.stack((x1, x2), dim=1)

        attn = torch.matmul(x / self.temperature, x.transpose(1, 2))
        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, x)

        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x = torch.cat((x1, x2), dim=-1)

        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "and_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x
