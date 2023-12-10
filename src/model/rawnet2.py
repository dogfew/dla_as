import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        take_abs=True,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        s3=False,
        min_low_hz=0,
        min_band_hz=0,
    ):
        super().__init__()

        if in_channels != 1:
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 1e-10
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            low_hz if s3 else self.to_mel(low_hz),
            high_hz if s3 else self.to_mel(high_hz),
            self.out_channels + 1,
        )
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        n = (self.kernel_size - 1) / 2.0
        self.n_ = nn.Parameter(
            math.tau * torch.arange(-n, 0).view(1, -1) / self.sample_rate,
            requires_grad=False,
        )
        self.window_ = nn.Parameter(
            0.54 - 0.46 * torch.cos(math.tau * n_lin / self.kernel_size),
            requires_grad=False,
        )
        self.take_abs = take_abs

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]
        band_pass_left = (
            (torch.sin(low @ self.n_) - torch.sin(high @ self.n_)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2.0 * band.view(-1, 1)
        band_pass_right = band_pass_left.flip(dims=[1])
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )
        band_pass = band_pass / (2.0 * band[:, None])
        out = F.conv1d(
            waveforms,
            band_pass.view(self.out_channels, 1, self.kernel_size),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        if self.take_abs:
            return out.abs()
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=True,
                padding=1,
            ),
        )
        self.downsample = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.max_pooling = nn.MaxPool1d(3)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                in_features=out_channels,
                out_features=out_channels,
            ),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(out_channels, -1)),
        )

    def forward(self, x):
        x = self.max_pooling(self.layers(x) + self.downsample(x))
        y = self.fc(x)
        return x * y + y


class RawNet2(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels_list: list[int],
        sinc_filter_size=1024,
        take_abs=True,
        min_low_hz=0,
        min_band_hz=0,
        gru_num_layers=3,
        s3=False,
        prebatchnorm_gru=True,
    ):
        super().__init__()
        first_channels = in_channels_list[0]
        last_channels = out_channels_list[-1]
        self.fixed_sinc_filters = nn.Sequential(
            SincConv_fast(
                out_channels=first_channels,
                kernel_size=sinc_filter_size,
                s3=s3,
                take_abs=take_abs,
                min_low_hz=min_low_hz,
                min_band_hz=min_band_hz,
            ),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(first_channels),
            nn.LeakyReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock(in_channels=in_channels, out_channels=out_channels)
                for in_channels, out_channels in zip(
                    in_channels_list, out_channels_list
                )
            ]
        )
        self.bn = (
            nn.Sequential(nn.BatchNorm1d(num_features=last_channels), nn.LeakyReLU())
            if prebatchnorm_gru
            else nn.Identity()
        )

        self.gru = nn.GRU(
            input_size=last_channels,
            hidden_size=1024,
            dropout=0.2,
            batch_first=True,
            num_layers=gru_num_layers,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Linear(in_features=1024, out_features=2),
        )

    def forward(self, waves, **kwargs):
        x = self.fixed_sinc_filters(waves.unsqueeze(dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.bn(x).transpose(2, 1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return {"logits": x}


if __name__ == "__main__":
    torch.manual_seed(124)
    net = RawNet2(
        in_channels_list=[20, 20, 20, 128, 128, 128],
        out_channels_list=[20, 20, 128, 128, 128, 128],
    )
    # for p in net.parameters():
    #     if p.requires_grad:
    #         print(p.numel())

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    inp = torch.randn((4, 64_000), dtype=torch.float32)
    print(net(inp))
# print(net)
