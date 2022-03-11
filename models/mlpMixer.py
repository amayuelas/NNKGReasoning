import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixer import MlpMixer

class ProjectionMixer(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(ProjectionMixer, self).__init__()
        self.mlpMixer = MlpMixer(patches=2,
                                 feature_length=entity_dim,
                                 num_classes=entity_dim,
                                 num_blocks=2,
                                 hidden_dim=200,
                                 tokens_mlp_dim=200,
                                 channels_mlp_dim=4
                                 )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        x = torch.cat((x1, x2), dim=-1)  # batch_size x feature_size x channels
        x = self.mlpMixer(x)
        return x


class AndMixer(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(AndMixer, self).__init__()
        self.mlpMixer = MlpMixer(patches=2,
                                 feature_length=entity_dim,
                                 num_classes=entity_dim,
                                 num_blocks=2,
                                 hidden_dim=200,
                                 tokens_mlp_dim=200,
                                 channels_mlp_dim=4
                                 )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        x = torch.cat((x1, x2), dim=-1)  # batch_size x feature_size x channels
        x = self.mlpMixer(x)
        return x


class OrMixer(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(OrMixer, self).__init__()
        self.mlpMixer = MlpMixer(patches=2,
                                 feature_length=entity_dim,
                                 num_classes=entity_dim,
                                 num_blocks=2,
                                 hidden_dim=200,
                                 tokens_mlp_dim=200,
                                 channels_mlp_dim=4
                                 )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        x = torch.cat((x1, x2), dim=-1)  # batch_size x feature_size x channels
        x = self.mlpMixer(x)
        return x