import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


class CDFEstimator(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super(CDFEstimator, self).__init__()

        self.first_layer = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.Tanh())
        self.hidden_layers = []

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.Tanh()))

        self.last_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], output_size), nn.Sigmoid())

    def forward(self, x):
        layers_outputs = []
        x = self.first_layer(x)
        layers_outputs.append(x)
        for layer in self.hidden_layers:
            x = layer(x)
            layers_outputs.append(x)
        x = self.last_layer(x)
        layers_outputs.append(x)
        return x, layers_outputs

