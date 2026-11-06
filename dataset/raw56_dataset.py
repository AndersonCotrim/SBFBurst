import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform

from admin.environment import env_settings
from data import processing, sampler
from dataset.base_rawburst_dataset import BaseRawBurstDataset


def load_txt(path):
    with open(path, 'r') as fh:
        out = [d.rstrip() for d in fh.readlines()]

    return out


class S23UltraRAW:
    """ Custom class for RAW images captured from Samsung Galaxy S23 Ultra """

    def __init__(self, im_raw, info, exif):
        self.im_raw = im_raw
        self.black_level = info['black_level']
        self.cam_wb = info['cam_wb']
        self.daylight_wb = info['daylight_wb']
        self.color_matrix = info['color_matrix']
        self.norm_factor = info['norm_factor']
        self.exif_data = exif
        self.info = info

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'color_matrix': self.color_matrix.tolist()}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data['Image Tag 0xC761'].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        return S23UltraRAW(im_raw, self.info, self.exif_data)

    def postprocess(self, return_np=True, norm_factor=None):
        raise NotImplementedError


class S23UltraRRGB:
    """ Custom class for RAW RGB images captured from Samsung Galaxy S23 Ultra """

    @staticmethod
    def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                                 no_white_balance=False):
        im = im * meta_data.get('norm_factor', 1.0)

        if not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb'][:3])[[0, 1, -1]].view(3, 1, 1) / \
                 torch.tensor(meta_data['cam_wb'][:3])[1]

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

    def __init__(self, im_raw, info, exif, p_black=False, p_norm=False):
        super(S23UltraRRGB, self).__init__()

        black_level = info['black_level']
        cam_wb = info['cam_wb']
        daylight_wb = info['daylight_wb']
        norm_factor = info['norm_factor']
        rgb_xyz_matrix = info['rgb_xyz_matrix']

        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix

        self.exif_data = exif
        self.info = info

        self.norm_factor = norm_factor
        self.p_black = p_black
        self.p_norm = p_norm

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'norm_factor': self.norm_factor}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level and not self.p_black:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize and not self.p_norm:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return S23UltraRRGB(im_raw, self.info, self.exif_data, self.p_black, self.p_norm)

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):
        self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
                                    mode='bilinear').squeeze(0)

    def postprocess(self, return_np=True):
        raise NotImplementedError

def shuffle(raw_data):
    canal_r = raw_data[0::2, 0::2]
    canal_g1 = raw_data[0::2, 1::2]
    canal_g2 = raw_data[1::2, 0::2]
    canal_b = raw_data[1::2, 1::2]
    return np.array([canal_r, canal_g1, canal_g2, canal_b])

class Raw56Dataset(BaseRawBurstDataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root=None, split='train', burst_size=14, sr_factor=4, random_sampling=False, crop_sz=80):
        root = env_settings().raw56data if root is None else root
        super().__init__('Raw56Dataset', root)

        self.split = split
        self.root = root
        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.random_sampling = random_sampling
        self.sr_factor = sr_factor

        print("Building Raw56 dataset for split: {}".format(split))
        self.burst_list = self._get_burst_list(split)
        print("Burst list size: ", len(self.burst_list))


    def _get_burst_list(self, split):
        seq_ids = np.array(load_txt('{}/split/{}.txt'.format(self.root, split)))
        return seq_ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': self.burst_size, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def get_burst(self, burst_id, im_ids, info=None):
        img_name = self.burst_list[burst_id]

        folder = Path(self.root, 'data', img_name)

        lr_images = np.load(folder / 'lr.npy')
        hr_image = np.load(folder / 'hr.npy')

        with open(folder / 'meta_lr.pkl', 'rb') as file:
            meta_lr = pickle.load(file)

        with open(folder / 'meta_hr.pkl', 'rb') as file:
            meta_hr = pickle.load(file)

        lr_images = np.array([shuffle(f) for f in lr_images])
        lr_images = lr_images[im_ids]

        start_idx = 0
        start_idy = 0
        if self.random_sampling:
            start_idx = random.randint(0, lr_images.shape[2] - self.crop_sz)
            start_idy = random.randint(0, lr_images.shape[3] - self.crop_sz)

        lr_images = lr_images[:, :, start_idx:start_idx+self.crop_sz, start_idy:start_idy+self.crop_sz]

        hr_size = self.crop_sz * 16
        start_idx_hr = start_idx * 16
        start_idy_hr = start_idy * 16

        hr_image = hr_image[start_idx_hr:start_idx_hr+hr_size, start_idy_hr:start_idy_hr+hr_size]

        hr_image = transform.resize(hr_image, (self.crop_sz*self.sr_factor*2, self.crop_sz*self.sr_factor*2, hr_image.shape[2]), anti_aliasing=True)


        hr_image = hr_image.transpose(2, 0, 1)

        lr_images = [S23UltraRAW(torch.from_numpy(f.astype(np.float32)), meta_lr['info'], meta_lr['exif']) for f in lr_images]
        gt = S23UltraRRGB(torch.from_numpy(hr_image.astype(np.float32)), meta_hr['info'], meta_hr['exif'])

        return lr_images, gt, info


def get_newbsr_val_set(burst_size=40, sr_factor=4):
    """ Get the BurstSR validation dataset """
    burstsr_dataset = Raw56Dataset(split='test', burst_size=burst_size, random_sampling=False, sr_factor=sr_factor)
    #burstsr_dataset.burst_list = burstsr_dataset.burst_list[:30]  # Limit to first 100 samples for validation

    processing_fn = processing.Raw56Processing(transform=None, random_flip=False,
                                               substract_black_level=False,
                                               crop_sz=80)
    # Train sampler and loader
    dataset = sampler.IndexedBurst(burstsr_dataset, burst_size=burst_size, processing=processing_fn)
    return dataset
