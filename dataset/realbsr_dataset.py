import os
import pickle

import cv2
import numpy as np
import torch

from admin.environment import env_settings
from dataset.base_rawburst_dataset import BaseRawBurstDataset


class SonyImage:
    """ Custom class for RAW images captured from Sony DSLR """

    @staticmethod
    def load_meta(path):
        with open(path, 'rb') as f:
            ret_dict = pickle.load(f)

        return ret_dict

    @staticmethod
    def generate_processed_image_channel4(im, meta_data, return_np=False, black_level_substracted=True,
                                          external_norm_factor=None,
                                          gamma=True, smoothstep=True, no_white_balance=False):
        im = im * meta_data.get('norm_factor', 16383.0)

        if not meta_data.get('black_level_subtracted', False) and not black_level_substracted:
            im = (im - torch.tensor(meta_data['black_level']).view(4, 1, 1))

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb']).view(4, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = torch.stack((im_out[0, :, :], im_out[1:3, :, :].mean(dim=0), im_out[3, :, :]), dim=0)
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out

    @staticmethod
    def generate_processed_image_channel3(im, meta_data, return_np=False, black_level_substracted=True,
                                          external_norm_factor=None,
                                          gamma=True, smoothstep=True, no_white_balance=False):
        im = im * meta_data.get('norm_factor', 16383.0)

        if not meta_data.get('black_level_subtracted', False) and not black_level_substracted:
            meta_data['black_level'] = torch.from_numpy(np.array(meta_data['black_level']))

            meta_data['black_level'] = torch.stack((meta_data['black_level'][0].float(),
                                                    meta_data['black_level'][1:3].float().mean(),
                                                    meta_data['black_level'][3].float()), dim=0)
            im = im - torch.tensor(meta_data['black_level']).view(3, 1, 1)

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            meta_data['cam_wb'] = torch.from_numpy(np.array(meta_data['cam_wb']))
            meta_data['cam_wb'] = torch.stack((meta_data['cam_wb'][0].float(), meta_data['cam_wb'][1:3].float().mean(),
                                               meta_data['cam_wb'][3].float()), dim=0)
            im = im * torch.tensor(meta_data['cam_wb']).view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)

        return im_out


class RealBSRDataset(BaseRawBurstDataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root=None, mode="RGB", split='train', aligned=False):
        """
        args:
            root - Path to root dataset directory
            split - Can be 'train' or 'val'
            seq_ids - (Optional) List of sequences to load. If specified, the 'split' argument is ignored.
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().rbsr_dir if root is None else root
        super().__init__('RGBRBSRDataset', root)

        if split == 'val':
            split = 'test'

        self.split = split
        self.mode = mode
        self.aligned = aligned
        self.root = os.path.join(root, mode)

        if mode == "RAW":
            if aligned:
                self.lr_path = os.path.join(self.root, '{}patch_aligned'.format(self.split))
            else:
                self.lr_path = os.path.join(self.root, '{}patch'.format(self.split))
            self.hr_path = os.path.join(self.root, '{}patch'.format(self.split))
        else:
            if aligned:
                self.lr_path = os.path.join(self.root, self.split, 'LR_aligned')
            else:
                self.lr_path = os.path.join(self.root, self.split, '{}patch'.format(self.split))
            self.hr_path = os.path.join(self.root, self.split, '{}patch'.format(self.split))

        self.burst_list = self._get_burst_list()

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.hr_path))
        return burst_list

    def get_burst_info(self, burst_id):

        if self.mode == 'RAW':
            burst_number2 = int(self.burst_list[burst_id].split('_')[-1])
            path = '{}/{}/MFSR_Sony_{:04d}_x4.pkl'.format(self.hr_path, self.burst_list[burst_id], burst_number2)
            burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id], 'meta_path': path}
        else:
            burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}

        return burst_info

    def _get_raw_image(self, burst_id, im_id):

        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])

        path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(self.lr_path, self.burst_list[burst_id], burst_number,
                                                                burst_number2, im_id)

        if self.mode == 'RGB':
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = torch.Tensor(im).permute(2, 0, 1)
        else:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            im = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1)  # [4, H, W] / [3, H, W]
            im = im / 16383.
            im = torch.Tensor(im)

        return im

    def _get_gt_image(self, burst_id):
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])

        if self.mode == 'RGB':
            path = '{}/{}/{}_MFSR_Sony_{:04d}_x4warp.png'.format(self.hr_path, self.burst_list[burst_id], burst_number,
                                                                 burst_nmber2)
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = torch.Tensor(im).permute(2, 0, 1)
        else:
            path = '{}/{}/{}_MFSR_Sony_{:04d}_x4_rgb.png'.format(self.hr_path, self.burst_list[burst_id], burst_number,
                                                                 burst_nmber2)

            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            im = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1)  # [4, H, W] / [3, H, W]
            im = im / 16383.
            im = torch.Tensor(im)

        return im

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        gt = self._get_gt_image(burst_id)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info
