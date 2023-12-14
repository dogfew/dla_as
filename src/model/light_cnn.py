import torch
import torch.nn as nn


class MaxFeatureMap(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

    def __repr__(self):
        return f"MFM({self.out_channels})"


class LightCNN(nn.Module):
    def __init__(self, first_dim=180, second_dim=600, use_dropout=True):
        # 180x600
        # 863x124
        super().__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim

        self.layers = nn.Sequential(
            *self._create_layer(1, 64, 5, batch_norm=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._create_layer(32, 64, 1),
            *self._create_layer(32, 96, 3, batch_norm=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            *self._create_layer(48, 96, 1),
            *self._create_layer(48, 128, 3, batch_norm=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self._create_layer(64, 128, 1),
            *self._create_layer(64, 64, 3),
            *self._create_layer(32, 64, 1),
            *self._create_layer(32, 64, 3, batch_norm=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.75) if use_dropout else nn.Identity(),
            nn.Linear((first_dim // 16) * (second_dim // 16) * 32, 160),
            MaxFeatureMap(80),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _create_layer(self, in_channels, out_channels, kernel_size, batch_norm=True):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            MaxFeatureMap(out_channels // 2),
        ] + [nn.BatchNorm2d(out_channels // 2)] * batch_norm

    def forward(self, mels, **kwargs):
        return {"logits": self.layers(mels.unsqueeze(1))}


def format_number(num):
    if num >= 1e6:
        formatted = f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        formatted = f"{num / 1e3:.1f}k"
    else:
        formatted = str(num)
    return formatted
