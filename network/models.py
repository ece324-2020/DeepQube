import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """A simple fully connected neural network with customizable
    input, output, and hidden layer dimensions, as well as activation.
    """
    def __init__(self, in_dim, out_dim, layers_dim, activation=F.relu):
        super(DQN, self).__init__()

        self.activation = activation
        self.head = nn.Linear(in_dim, layers_dim[0])
        self.body = nn.ModuleList([
            nn.Linear(layers_dim[i - 1], layers_dim[i])
            for i in range(1, len(layers_dim))])
        self.tail = nn.Linear(layers_dim[-1], out_dim)

    def forward(self, x):
        # TODO: Should we use dropout?
        x = self.activation(self.head(x))
        for layer in self.body:
            x = self.activation(layer(x))
        return self.tail(x)

class Minimal(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation=F.relu):
        super(Minimal, self).__init__()

        self.activation = activation
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

