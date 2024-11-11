import torch.nn as nn

class Model(nn.Module):
    """
    Base model to use in the Deep Q Network
    """
    def __init__(self, width, height, hidden_dim=128):
        super().__init__()
        self.width = width
        self.height = height
        self.fc = nn.Sequential(
            nn.Linear(width * height, hidden_dim),
            nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, width * height)
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

