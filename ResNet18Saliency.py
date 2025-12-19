import torch
import torch.nn as nn
import torchvision.models as models
import os


LOCAL_WEIGHT_PATH = r"resnet18-f37072fd.pth"
class ResNet18Saliency(nn.Module):
    def __init__(self, pretrained=True):  # <--- 你的核心要求：不使用预训练权重
        super().__init__()
        # 加载 ResNet18 结构，但不加载权重
        # resnet = models.resnet18(pretrained=pretrained)

        resnet = models.resnet18(weights=None)
        #
        # # 2. 手动加载本地权重
        if pretrained:
            if os.path.exists(LOCAL_WEIGHT_PATH):
                print(f"正在加载本地预训练权重: {LOCAL_WEIGHT_PATH}")
                # 加载权重字典
                state_dict = torch.load(LOCAL_WEIGHT_PATH, map_location='cpu')
                # 将权重载入模型
                resnet.load_state_dict(state_dict)
            else:
                print(f"⚠️ 警告: 未找到本地权重文件: {LOCAL_WEIGHT_PATH}")
                print(">>> 模型将使用【随机初始化】权重进行训练 (效果会变差)")
        else:
            print("注意：正在使用随机初始化权重 (Training from Scratch)...")

        # 编码器 (Encoder)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # 解码器 (Decoder)
        # self.decoder5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.decoder4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)
        # self.decoder3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        # self.decoder2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2)
        # self.decoder1 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=2, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder5 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # d5 = self.decoder5(e5)
        # d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        # d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        # d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        # out = self.decoder1(torch.cat([d2, e1], dim=1))
        d5 = self.decoder5(self.up(e5))
        if d5.size()[2:] != e4.size()[2:]: d5 = torch.nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear',
                                                                                align_corners=True)

        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d4 = self.up(d4)
        if d4.size()[2:] != e3.size()[2:]: d4 = torch.nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear',
                                                                                align_corners=True)

        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d3 = self.up(d3)
        if d3.size()[2:] != e2.size()[2:]: d3 = torch.nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear',
                                                                                align_corners=True)

        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d2 = self.up(d2)
        if d2.size()[2:] != e1.size()[2:]: d2 = torch.nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear',
                                                                                align_corners=True)

        out = self.decoder1(torch.cat([d2, e1], dim=1))
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return self.sigmoid(out)
