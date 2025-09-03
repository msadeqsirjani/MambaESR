import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64, num_layers=3):
        super().__init__()
        layers = []

        # Initial layer
        layers.append(nn.Conv2d(in_channels, features, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        mult = 1
        for i in range(1, num_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        features * mult_prev, features * mult, 4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(features * mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        # Final layers
        layers.extend(
            [
                nn.Conv2d(features * mult, features * 8, 4, stride=1, padding=1),
                nn.BatchNorm2d(features * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features * 8, 1, 4, stride=1, padding=1),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
