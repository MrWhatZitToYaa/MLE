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
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            #nn.Softmax(1)
        )
        weights = np.random.rand(len(ACTIONS))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #print(x)
        output = self.model(torch.Tensor(x))
        
        return output
    
    
