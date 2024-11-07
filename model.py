"""

State: Tensor of shape (batch_size, width, height) representing the current board states
Actions: Tensor of shape (batch_size, 2) representing the x, y coordinates of the click


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, width, height):
        super(Model, self).__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU()
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * width * height, 128),
            nn.ReLU(),
            nn.Linear(128, width * height)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64 * self.width * self.height)
        x = self.fc(x)
        return x

