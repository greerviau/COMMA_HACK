from torch import nn

class QNet(nn.Module):   
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 10, 3)
        self.conv2 = nn.Conv2d(10, 50, 3)
        self.conv3 = nn.Conv2d(50, 50, 3)
        self.conv4 = nn.Conv2d(50, 50, 3)
        self.conv5 = nn.Conv2d(50, 1, 3)

        self.lin = nn.Linear(200, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.flatten(x)
        x = self.lin(x).tanh()
        return x
