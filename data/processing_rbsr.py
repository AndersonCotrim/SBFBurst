import random

import torch

from data.processing import BaseProcessing


class Augment_RGB_torch:
    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor

    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1, -2])
        return torch_tensor

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1, -2])
        return torch_tensor

    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1, -2])
        return torch_tensor

    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor

    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1, -2])).flip(-2)
        return torch_tensor


augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def get_crop(img, r1, r2, c1, c2):
    im_raw = img[:, r1:r2, c1:c2]
    return im_raw


class RBSRProcessing(BaseProcessing):
    """ The processing class used for training on BurstSR dataset. """

    def __init__(self, crop_sz=64, *args, **kwargs):
        """
        args:
            crop_sz - Size of the extracted crop
            substract_black_level - Boolean indicating whether to subtract the black level from the sensor data
            white_balance - Boolean indicating whether to apply white balancing provided by the camera
            random_flip - Boolean indicating whether to apply random flips to sensor data
            noise_level - The parameters of the synthetic noise added to sensor data. If None, no noise is added
            random_crop - Boolean indicating whether to perform random cropping (True) or center cropping (False)
        """
        super().__init__(*args, **kwargs)
        self.crop_sz = crop_sz

    def __call__(self, data):
        frames = data['frames']
        gt = data['gt']

        # Extract crop if needed
        if self.crop_sz is not None and frames[0].shape[-1] != self.crop_sz:
            r1 = random.randint(0, frames[0].shape[-2] - self.crop_sz)
            c1 = random.randint(0, frames[0].shape[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            scale_factor = gt.shape[-1] // frames[0].shape[-1]

            # print(scale_factor)
            frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
            gt = get_crop(gt, scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        apply_trans = transforms_aug[random.getrandbits(3)]
        frames = [getattr(augment, apply_trans)(im) for im in frames]
        gt = getattr(augment, apply_trans)(gt)

        burst = torch.stack(frames, dim=0)

        data['frames'] = burst
        data['gt'] = gt

        return data
