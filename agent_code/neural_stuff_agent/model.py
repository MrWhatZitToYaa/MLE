import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .state_to_feature import *
from .definitions import *
from .state_to_feature import *
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        self.model = nn.Sequential()
        #channels = [input_channels, 64, 32, num_classes]
        self.model = nn.Sequential(
            nn.Linear(input_channels, 64),  # def 2048, 512, 15*15 is image size after conv
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=0)
        )
        '''
        for i in range(len(channels)-1):
            self.model.add_module(
                f'linear{i}',
                nn.Linear(channels[i], channels[i+1])
            )
            self.model.add_module(
                f'relu{i}',
                nn.ReLU() )
        self.model.add_module('Softmax', nn.Softmax(dim=0.5))
        '''
        weights = np.random.rand(len(ACTIONS))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # shape of images: (batch_size, channels, height, width)
        #x = x.flatten()
        output = self.model(torch.Tensor(x))
        #output = F.softmax(x)
        
        return output
    
    
