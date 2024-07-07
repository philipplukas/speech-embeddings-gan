import torch
import torch.nn as nn

def dense(in_features, out_features, bn=True):
    """Custom dense layer for simplicity."""
    layers = []
    layers.append(nn.Linear(in_features, out_features, bias=False))
    if bn:
        layers.append(nn.BatchNorm1d(out_features))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator for transferring from embedding vector A to B"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super(G12, self).__init__()
        self.fc1 = dense(input_dim, hidden_dim)
        self.fc2 = dense(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x), 0.05)
        out = F.leaky_relu(self.fc2(out), 0.05)
        out = F.tanh(self.fc3(out))
        return out



class Discriminator(nn.Module):
    """Discriminator for embedding vector A"""
    def __init__(self, input_dim=128, hidden_dim=256):
        super(D1, self).__init__()
        self.fc1 = dense(input_dim, hidden_dim, bn=False)
        self.fc2 = dense(hidden_dim, hidden_dim)
        n_out = 1
        self.fc3 = nn.Linear(hidden_dim, n_out)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x), 0.05)
        out = F.leaky_relu(self.fc2(out), 0.05)
        out = self.fc3(out).squeeze()
        return out


