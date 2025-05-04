
"""
models.py - generator & discriminator factory functions.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_generator(encoder_name: str = "resnet18", pretrained: bool = True):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,
        activation=None
    )

class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator."""
    def __init__(self, in_channels: int = 6, features: int = 64):
        super().__init__()
        def blk(cin, cout, stride=2, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, stride, 1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            blk(in_channels, features, bn=False),
            blk(features, features*2),
            blk(features*2, features*4),
            blk(features*4, features*8),
            blk(features*8, features*8, stride=1),
            nn.Conv2d(features*8, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, gray, color):
        x = torch.cat([gray, color], 1)
        return self.net(x)
