from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters

        # First convolutional layer of the block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)

        # Second convolutional layer of the block
        self.conv2 = nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        # Third convolutional layer of the block
        self.conv3 = nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

        # Shortcut connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f3, kernel_size=1),
            nn.BatchNorm2d(f3)
        )

        # Non-linear activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Keep a copy of the input (for the shortcut connection)
        shortcut = self.shortcut(x)

        # Main path of the block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Add the output of the main path and the shortcut connection
        x += shortcut

        # Apply activation function
        x = self.relu(x)

        return x
    


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters

        # First convolutional layer of the block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)

        # Second convolutional layer of the block
        self.conv2 = nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        # Third convolutional layer of the block
        self.conv3 = nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

        # Non-linear activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Keep a copy of the input (for the shortcut connection)
        shortcut = x

        # Main path of the block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Add the output of the main path and the shortcut connection
        x += shortcut

        # Apply activation function
        x = self.relu(x)

        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResBlock, self).__init__()

        # Convolutional block
        self.conv_block = ConvBlock(in_channels, filters)

        # Identity block 1
        self.id_block1 = IdentityBlock(filters[-1], filters)

        # Identity block 2
        self.id_block2 = IdentityBlock(filters[-1], filters)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.id_block1(x)
        x = self.id_block2(x)
        return x
    


class ModResNet(nn.Module):
    def __init__(self):
        super(ModResNet, self).__init__()

        # Initial layers
        self.pad = nn.ZeroPad2d(padding=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Blocks
        self.res_block1 = ResBlock(in_channels=64, filters=[64, 64, 256])
        self.res_block2 = ResBlock(in_channels=256, filters=[128, 128, 512])

        # Average Pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 30)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        # Initial layers
        x = self.pad(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual Blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Average Pooling layer
        x = self.avgpool(x)

        # Flattening for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x