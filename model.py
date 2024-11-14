import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(height, 3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(height, 3), stride=1, padding="same"),
                nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * width * height,  256),
            nn.ReLU(),
            nn.Linear(256, width * height)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.view(-1, 1, self.height, self.width)
        x = self.conv(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc(x)
        return x

