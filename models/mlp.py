import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(ProjectionMLP, self).__init__()
        self.n_layers = n_layers
        for i in range(1, self.n_layers+1):
            setattr(self, "proj_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers+1):
            x = F.relu(getattr(self, "proj_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class AndMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(AndMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "and_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "and_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class OrMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(OrMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "or_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "or_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class NotMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(NotMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "not_layer_{}".format(i), nn.Linear(entity_dim, entity_dim))
        self.last_layer = nn.Linear(entity_dim, entity_dim)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "not_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x
