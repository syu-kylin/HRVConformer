##################################################
# ResNet for 1D HRV data.
# Reference:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
##################################################


import torch
import torch.nn as nn
from torchsummary import summary
import logging

import timm.models.resnet

class ResidualBlock1D(nn.Module):
    """A single residual block with optional dilation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation  # Keep same length

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample  # If needed for dimension matching

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu2(out)
        return out
    
class BottleneckBlock1D(nn.Module):
    """Bottleneck block for 1D ResNet with 1x1 -> 3x3 -> 1x1 convolutions."""
    expansion = 4  # Expands channels after the final 1x1 conv

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock1D, self).__init__()
        bottleneck_channels = out_channels // self.expansion  # Reduce dimensionality

        # 1x1 Convolution (Reduce dimensions)
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        # 3x3 Convolution (Feature extraction)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        # 1x1 Convolution (Restore dimensions)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Adjust residual connection if needed

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # Skip connection
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """ResNet for 1D HRV data using bottleneck blocks."""
    def __init__(self, block, in_channels=1, num_classes=2, layers=[3, 4, 6, 3], drop_rate=0):  # ResNet-50 default
        super(ResNet1D, self).__init__()
        self.inplanes = 64

        # Initial Convolution Layer
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Create ResNet Layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Reduces to (batch, channels, 1)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(drop_rate)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Creates a ResNet block with multiple bottleneck units."""
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(block(self.inplanes, out_channels, stride=stride, downsample=downsample))
        self.inplanes = out_channels  # Update the inplanes for the next blocks

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        return x


def resnet1d18(in_channels, num_classes, drop_rate=0):
    """ResNet-18 model for 1D HRV data."""
    return ResNet1D(ResidualBlock1D, in_channels=in_channels, num_classes=num_classes, layers=[2, 2, 2, 2], drop_rate=drop_rate)

def resnet1d34(in_channels, num_classes, drop_rate=0):
    """ResNet-34 model for 1D HRV data."""
    return ResNet1D(ResidualBlock1D, in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 6, 3], drop_rate=drop_rate)

def resnet1d50(in_channels, num_classes, drop_rate=0):
    """ResNet-50 model for 1D HRV data."""
    return ResNet1D(BottleneckBlock1D, in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 6, 3], drop_rate=drop_rate)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Instantiate Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D(BottleneckBlock1D, in_channels=1, num_classes=2, layers=[3, 4, 6, 3])  # ResNet-50
    model.to(device)
    logging.info(model)

    # Example Input: Batch of 8 HRV sequences, each with 1 channel and 1200 time steps
    x = torch.randn(8, 1, 1200).to(device)
    output = model(x)
    logging.info(output.shape)  # Expected output: (8, 2)

    # Model Summary
    summary(model, input_size=(1, 1200))