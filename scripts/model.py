import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            
            nn.Conv2d(f2, f3, kernel_size=1, stride=1),
            nn.BatchNorm2d(f3),
        )

        self.shortcut_path = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(f3),
        )

    def forward(self, x):
        shortcut = self.shortcut_path(x)
        x = self.main_path(x)
        x += shortcut
        return F.relu(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            
            nn.Conv2d(f2, f3, kernel_size=1),
            nn.BatchNorm2d(f3),
        )

    def forward(self, x):
        shortcut = x
        x = self.main_path(x)
        x += shortcut
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.pad = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.res2 = ConvBlock(64, [64, 64, 256], stride=1)
        self.id21 = IdentityBlock(256, [64, 64, 256])
        self.id22 = IdentityBlock(256, [64, 64, 256])
        self.res3 = ConvBlock(256, [128, 128, 512], stride=2)
        self.id31 = IdentityBlock(512, [128, 128, 512])
        self.id32 = IdentityBlock(512, [128, 128, 512])
        self.avgpool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(512, 4096)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4096, 2048)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(2048, 30)  # Change the output size to match the number of keypoints*2

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.id21(x)
        x = self.id22(x)
        x = self.res3(x)
        x = self.id31(x)
        x = self.id32(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)  # We don't apply activation function here as we want to predict continuous coordinates.
        return x