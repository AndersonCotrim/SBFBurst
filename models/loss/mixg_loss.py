import kornia
import torch.nn.functional as F
from torch import nn


def mixed_gradient_loss(predicted_image, ground_truth_image, lambda_value, reduction="mean"):
    mse_loss = F.l1_loss(predicted_image, ground_truth_image, reduction=reduction)

    predicted_gradient = kornia.filters.sobel(predicted_image)
    ground_truth_gradient = kornia.filters.sobel(ground_truth_image)

    gradient_mse_loss = F.mse_loss(predicted_gradient, ground_truth_gradient, reduction=reduction)

    mixge_loss = mse_loss + lambda_value * gradient_mse_loss

    return mixge_loss


class PixelWiseMGLError(nn.Module):

    def __init__(self, boundary_ignore=None):
        self.boundary_ignore = boundary_ignore
        super().__init__()

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]

        if valid is None:
            err = mixed_gradient_loss(pred, gt, 0.01)
        else:
            err = mixed_gradient_loss(pred, gt, 0.01, reduction="none")
            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err
