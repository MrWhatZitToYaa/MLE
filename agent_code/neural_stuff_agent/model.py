import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .state_to_feature import *
from .definitions import *

class Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        self.model = nn.Sequential()
        channels = [input_channels, 64, 32, num_classes]

        for i in range(len(channels)-1):
            self.model.add_module(
                f'conv{i}',
                nn.Linear(channels[i], channels[i+1])
            )

    def forward(self, x):
        # shape of images: (batch_size, channels, height, width)
        x = x.flatten()
        output = self.model(state_to_features(x))

        return output
    
    
