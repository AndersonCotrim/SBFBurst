import torch
import torch.nn as nn


def combine_bayer_channels(bayer_image):
    # Assuming bayer_image is a PyTorch tensor with shape (B, C, H, W),
    # where B is the batch size, C is the number of channels (should be 4 for Bayer),
    # H is the height, and W is the width.

    # Split the Bayer channels
    red_channel = bayer_image[:, 0:1, :, :]
    green1_channel = bayer_image[:, 1:2, :, :]
    green2_channel = bayer_image[:, 2:3, :, :]
    blue_channel = bayer_image[:, 3:4, :, :]

    # Interleave the channels to create RGGB pattern

    even_rows = torch.cat((red_channel.unsqueeze(4), green1_channel.unsqueeze(4)), dim=4)
    even_rows = even_rows.view(*even_rows.shape[:3], -1)

    odd_rows = torch.cat((green2_channel.unsqueeze(4), blue_channel.unsqueeze(4)), dim=4)
    odd_rows = odd_rows.view(*odd_rows.shape[:3], -1)

    rggb_image = torch.cat((even_rows.unsqueeze(3), odd_rows.unsqueeze(3)), dim=3)
    rggb_image = rggb_image.view(*rggb_image.shape[:2], -1, rggb_image.shape[-1])

    return rggb_image


class MCFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MCFE, self).__init__()
        self.phase_branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1,
                      dilation=1, groups=1, bias=True)
            for _ in range(4)
        ])
        self.offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def forward(self, x):
        phase_outputs = []
        for branch, offset in zip(self.phase_branches, self.offsets):
            x_offset = x[:, :, offset[0]:, offset[1]:]
            phase_outputs.append(branch(x_offset))

        # Combine phase outputs (up to you how you want to combine them)
        combined_features = torch.cat([p.unsqueeze(2) for p in phase_outputs], dim=2)

        result = combine_bayer_channels(combined_features.view(-1, *combined_features.shape[2:]))

        result = result.view(*combined_features.shape[:2], *result.shape[-2:])
        return result
