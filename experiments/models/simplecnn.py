import torch

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(256, 64)
        self.relu3 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(120, 84)
        # self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        # y = self.fc2(y)
        # y = self.relu4(y)
        y = self.fc3(y)
        return y