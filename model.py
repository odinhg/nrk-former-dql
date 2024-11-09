import torch.nn as nn

class Model(nn.Module):
    """
    Base model to use in the Deep Q Network
    """
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.fc = nn.Sequential(
            nn.Linear(width * height, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, width * height)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Normalize the input
        x = x / (self.width * self.height)
        x = self.fc(x)
        return x

