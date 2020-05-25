"""
Adaption of ResNet to fit the Idsgame input
"""
import torch
import torchvision.models as models


class IdsGameResNet(torch.nn.Module):

    def __init__(self, in_channels=1, output_dim=44):
        super(IdsGameResNet, self).__init__()

        # bring resnet
        self.model = models.resnet18(pretrained=False, num_classes=output_dim, norm_layer=torch.nn.InstanceNorm2d)

        # original definition of the first layer on the resnet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.conv1 = torch.nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)