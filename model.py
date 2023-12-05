import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        input_size = np.array(x.shape[-2:])
        if any(self.output_size > input_size):
            padding_height = max(self.output_size[0] - input_size[0], 0)
            padding_width = max(self.output_size[1] - input_size[1], 0)
            pad = nn.ZeroPad2d((0, padding_width, 0, padding_height))
            x = pad(x)

        input_size = np.array(x.shape[-2:])
        stride_size = np.floor(input_size / self.output_size).astype(np.int32)
        stride_size = np.clip(stride_size, a_min=1, a_max=None)
        kernel_size = input_size - (self.output_size - 1) * stride_size
        kernel_size = np.clip(kernel_size, a_min=1, a_max=None)

        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


def _make_deconv_layer(num_layers):
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


class SignatureCenterNet(nn.Module):
    def __init__(self, target_height=936, target_width=662):
        super(SignatureCenterNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.deconv_layers = _make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.adaptive_pool = AdaptiveAvgPool2dCustom((target_height, target_width))

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        heatmap = self.final_layer(x)
        heatmap = self.adaptive_pool(heatmap)
        return heatmap
