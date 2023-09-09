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
        self.model = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #print(x)
        output = self.model(torch.Tensor(x))
        
        return output
    
    
