import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .state_to_feature import *
from .definitions import *
from .state_to_feature import *
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels, fc1, fc2, num_classes):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(input_channels, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, num_classes),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.MSELoss()
        #self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        #self.to(self.device)

    def forward(self, x):
        #print('before', x)
        if type(x) == dict:
            x = state_to_features(x)
        #print('after', x)
        #print(x.shape)
        output = self.model(torch.Tensor(x))
        return output
    
    
