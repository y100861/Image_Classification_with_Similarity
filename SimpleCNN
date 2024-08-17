# -*- coding: utf-8 -*-

# 4 Layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(128)

        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        # Layer 1 (Conv - BatchNorm - ReLU - MaxPool - AvgPool)
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        # Layer 2
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))

        # Layer 3
        x = self.maxpool(F.relu(self.bn3(self.conv3(x))))

        # Layer 4
        x = self.maxpool(F.relu(self.bn4(self.conv4(x))))

        # AvgPool
        x = self.global_pool(x)

        return x


# Classifier
class testModel(nn.Module):
    def __init__(self, n_classes):
        super(testModel, self).__init__()
        self.feat_extractor = SimpleCNN()
        self.classifier = nn.Linear(128, n_classes)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.MaxPool2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.feat_extractor(x)
        feat = torch.flatten(feat, 1)
        output = self.classifier(feat)

        return output
