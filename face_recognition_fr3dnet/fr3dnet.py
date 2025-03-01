import torch
import torch.nn as nn

class FR3DNet(nn.Module):
    def __init__(self, num_classes=1853):
        super(FR3DNet, self).__init__()
        
        # Layer 1: Conv -> ReLU
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True)
        self.relu1_1 = nn.ReLU()
        
        # Layer 2: Conv -> ReLU
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True)
        self.relu1_2 = nn.ReLU()
        
        # Layer 3: Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Layer 4: Conv -> ReLU
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU()
        
        # Layer 5: Conv -> ReLU
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_2 = nn.ReLU()
        
        # Layer 6: Max Pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Layer 7: Conv -> ReLU
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_1 = nn.ReLU()
        
        # Layer 8: Conv -> ReLU
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_2 = nn.ReLU()
        
        # Layer 9: Conv -> ReLU
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_3 = nn.ReLU()
        
        # Layer 10: Max Pooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Layer 11: Conv -> ReLU
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4_1 = nn.ReLU()
        
        # Layer 12: Conv -> ReLU
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4_2 = nn.ReLU()
        
        # Layer 13: Conv -> ReLU
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4_3 = nn.ReLU()
        
        # Layer 14: Max Pooling
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Layer 15: Conv -> ReLU
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5_1 = nn.ReLU()
        
        # Layer 16: Conv -> ReLU
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5_2 = nn.ReLU()
        
        # Layer 17: Conv -> ReLU
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5_3 = nn.ReLU()
        
        # Layer 18: Max Pooling
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Layer 19: Conv (Fully Connected)
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu6 = nn.ReLU()
        
        # Layer 20: Conv (Fully Connected)
        self.fc7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu7 = nn.ReLU()
        
        # Layer 21: Conv (Fully Connected)
        self.fc8 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass through first convolution block
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        
        # Max pooling layer
        x = self.pool1(x)
        
        # Second convolution block
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        
        # Max pooling layer
        x = self.pool2(x)
        
        # Third convolution block
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        
        # Max pooling layer
        x = self.pool3(x)
        
        # Fourth convolution block
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        
        # Max pooling layer
        x = self.pool4(x)
        
        # Fifth convolution block
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        
        # Max pooling layer
        x = self.pool5(x)
        
        # Fully connected layers
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        
        # Softmax
        x = self.softmax(x)
        
        return x