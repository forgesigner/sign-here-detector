import torch.nn as nn
import torchvision.models as models


class SignatureCenterNet(nn.Module):
    def __init__(self, target_height=936, target_width=662):
        super(SignatureCenterNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_height, target_width))

    def _make_deconv_layer(self, num_layers):
        layers = []
        in_channels = 512
        out_channels = 256
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = out_channels // 2

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        heatmap = self.final_layer(x)
        heatmap = self.adaptive_pool(heatmap)
        return heatmap
