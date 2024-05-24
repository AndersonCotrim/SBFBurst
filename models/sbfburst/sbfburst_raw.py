from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sbfburst.common.defconv import DeformableAlignment
from models.sbfburst.common.flow_warp import flow_warp
from models.sbfburst.common.fusion import WeightedSum
from models.sbfburst.common.mfce import MCFE, combine_bayer_channels
from models.sbfburst.common.spynet import SPyNet
from models.sbfburst.common.sr_backbone_utils import ResidualBlocksWithInputConv
from models.sbfburst.common.upsample import PixelShufflePack


class SBFBurstRAW(nn.Module):
    def __init__(self, mid_channels=64, encoder_blocks=40):

        """
        Initialize the model with the given parameters.
        Args:
            mid_channels (int): The number of mid channels.
            encoder_blocks (int): The number of encoder blocks.
        """

        super().__init__()

        self.mid_channels = mid_channels

        # alignment components
        self.mfce = nn.Sequential(
            MCFE(1, 4),
            MCFE(4, 16),
        )
        self.conv1x1_f = nn.Conv2d(16, 3, 1, stride=1, padding=0)
        self.spynet = SPyNet(pretrained='./pretrained_networks/spynet_20210409-c6c1bd09.pth')
        self.deformable_alignment = DeformableAlignment(mid_channels, mid_channels, 3, padding=1, deform_groups=8,
                                                        max_residue_magnitude=10)

        # preencoder module
        self.preencoder = ResidualBlocksWithInputConv(4, mid_channels, 5)

        # encoder module
        self.encoder = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, encoder_blocks)

        # fusion module
        self.fusion = WeightedSum(mid_channels, offset_feat_dim=64, num_offset_feat_extractor_res=1,
                                  num_weight_predictor_res=3, offset_modulo=1.0, use_offset=True, ref_offset_noise=0.0,
                                  softmax=True, use_base_frame=True)

        # decoder module
        self.initial_conv_upsampling = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, 5)

        self.upsamplers = nn.Sequential(PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                        PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                        PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3))

        self.skippers_up = nn.Sequential(PixelShufflePack(4, mid_channels, 2, upsample_kernel=3),
                                         PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                         PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.final_conv_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(mid_channels, 3, 3, 1, 1)
        )

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path)['net']
        updated_dict = OrderedDict()

        # for optimized compiled models
        for key, value in checkpoint.items():
            new_key = key.replace('_orig_mod.', '', 1)
            updated_dict[new_key] = value

        checkpoint = updated_dict
        self.load_state_dict(checkpoint)

    def compute_flow(self, input_burst):

        input_burst = input_burst.clone()

        input_shape = input_burst.shape
        reshaped_images = input_burst.view(-1, *input_burst.shape[-3:])

        # Apply the mosaicked feature extractor and combine the channels
        combined_images = combine_bayer_channels(reshaped_images)
        feature_extracted_images = self.mfce(combined_images)

        # Combine the features and reshape back to the original shape
        combined_features = self.conv1x1_f(feature_extracted_images)
        reshaped_features = combined_features.view(input_shape[0], input_shape[1], -1, input_shape[3] * 2,
                                                   input_shape[4] * 2)

        # Compute the optical flow between the reference and other images
        batch_size, num_images, num_channels, height, width = reshaped_features.size()
        other_images = reshaped_features[:, 1:, :, :, :].reshape(-1, num_channels, height, width)
        ref_images = reshaped_features[:, :1, :, :, :].repeat(1, num_images - 1, 1, 1, 1).reshape(-1, num_channels,
                                                                                                  height, width)
        optical_flows = self.spynet(ref_images, other_images)

        # Downsample the optical flows and reshape to match the input shape
        downsampled_flows = F.avg_pool2d(optical_flows, 2, 2) / 2.
        reshaped_flows = downsampled_flows.view(batch_size, num_images - 1, 2, int(height / 2), int(width / 2))

        return reshaped_flows

    def compute_encoder_and_fusion(self, features, offsets):

        # Prepare the reference and other features
        reference_features = features[:, :1, :, :, :].repeat(1, features.shape[1], 1, 1, 1)
        other_features = features

        # Concatenate the reference and other features
        features = torch.cat((reference_features, other_features), dim=2)

        # Reshape the features for the encoder
        batch_size, num_bursts, num_features, height, width = features.size()
        features = features.view(-1, num_features, height, width)

        # Pass the features through the encoder
        encoded_features = self.encoder(features)
        encoded_features = encoded_features.view(batch_size, num_bursts, self.mid_channels, height, width)

        # Prepare the reference and other features from the encoded features
        reference_features = encoded_features[:, :1, :, :, :].repeat(1, num_bursts - 1, 1, 1, 1)
        other_features = encoded_features[:, 1:, :, :, :].contiguous()

        # Reshape the offsets to match the features
        offsets = offsets.view(batch_size, num_bursts - 1, 2, height, width)

        # Perform the merging operation
        fused = self.fusion({'ref_feat': reference_features, 'oth_feat': other_features, 'offsets': offsets})

        # Return the fused encoding
        return fused['fused_enc']

    def decoder(self, fused_encoder_feats, input_burst, base_feature):

        # Prepare the image levels for skip connections **base frame guidance**
        image_levels = [input_burst[:, 0, :, :, :]]
        for i in range(len(self.skippers_up)):
            image_levels.append(self.skippers_up[i](image_levels[i]))

        # Concatenate the base features and high resolution features  **base frame guidance**
        fused_encoder_feats = torch.cat([base_feature, fused_encoder_feats], dim=1)

        # Pass the high resolution features through the initial convolution before upsampling
        fused_encoder_feats = self.initial_conv_upsampling(fused_encoder_feats)

        # Perform the upsampling and add the skip connections
        for i in range(len(self.upsamplers)):
            fused_encoder_feats = self.leaky_relu(self.upsamplers[i](fused_encoder_feats)) + image_levels[i + 1]

        # Pass the upsampled features through the final convolution block
        fused_encoder_feats = self.final_conv_block(fused_encoder_feats)

        return fused_encoder_feats

    def forward(self, input_burst):
        # Get the dimensions of the input images
        batch_size, num_images, num_channels, height, width = input_burst.size()

        # Pass the images through the Preencoder
        features = self.preencoder(input_burst.view(-1, num_channels, height, width))

        # Reshape the preencoded features to match the input shape
        features = features.view(batch_size, num_images, -1, height, width)

        # Prepare the reference and other features
        reference_features = (features[:, :1, :, :, :].repeat(1, num_images - 1, 1, 1, 1)
                              .view(-1, *features.shape[-3:]))
        other_features = features[:, 1:, :, :, :].contiguous().view(-1, *features.shape[-3:])

        ## Alignment
        # Compute the optical flow between the reference and other images
        optical_flows = self.compute_flow(input_burst)
        optical_flows = optical_flows.view(-1, 2, *features.shape[-2:])

        # Warp the other features using the optical flow
        warped_other_features = flow_warp(other_features, optical_flows.permute(0, 2, 3, 1))

        # Refine the alignment of the other features with the reference features
        aligned_other_features = self.deformable_alignment(other_features, reference_features, warped_other_features,
                                                           optical_flows)

        # Reshape the aligned other features to match the input shape
        aligned_other_features = aligned_other_features.view(batch_size, num_images - 1, -1, height, width)

        # Prepare the reference features from the preencoded features
        reference_features = reference_features.view(batch_size, num_images - 1, -1, height, width)[:, :1, :, :, :]

        # Concatenate the reference and other features, **base frame guidance**
        encoder_input = torch.cat((reference_features, aligned_other_features), dim=1)

        # Compute the encoder and fusion
        fused_encoder_out = self.compute_encoder_and_fusion(encoder_input, optical_flows)

        # Return the upsampled high resolution image
        reference_features = reference_features.view(batch_size, -1, height, width)
        return self.decoder(fused_encoder_out, input_burst, reference_features)


if __name__ == '__main__':
    input_tensor = torch.randn(2, 8, 4, 48, 48).to('cuda')
    model = SBFBurstRAW()
    model = model.to('cuda')

    # Get the output tensor from the model
    output_tensor = model(input_tensor)

    # # Print the shapes of input and output tensors
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor[0].shape)
