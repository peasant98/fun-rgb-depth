import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEstimationNet(nn.Module):
    def __init__(self):
        super(DepthEstimationNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Upsample layers
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=2)

        
        # Decoder convolutional layers
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Decoder
        # x = F.relu(self.conv5(self.up1(x)))
        x = F.relu(self.conv6(self.up2(x)))

        x = F.relu(self.conv7(self.up3(x)))
        x = F.relu(self.conv8(self.up4(x)))

        return x

# Create the network

if __name__ == '__main__':
    net = DepthEstimationNet()
    print(net)
