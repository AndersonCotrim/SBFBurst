import os

import cv2
import numpy as np
import torch
import tqdm

from admin.environment import env_settings
from data.postprocessing_functions import SimplePostProcess
from dataset.synthetic_burst_val_set import SyntheticBurstVal
from models.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from models.sbfburst.sbfburst_raw import SBFBurstRAW
from utils.display_utils import generate_formatted_report

if __name__ == '__main__':
    base_results_dir = env_settings().save_predictions_path
    dataset = SyntheticBurstVal()
    process_fn = SimplePostProcess(return_np=True)

    net = SBFBurstRAW()
    net.load_checkpoint("./pretrained_networks/pretrained_synthetic.pth.tar")
    net.to('cuda').train(False)

    device = 'cuda'
    boundary_ignore = 40

    metrics_all = {
        'psnr': PSNR(boundary_ignore=boundary_ignore),
        'ssim': SSIM(boundary_ignore=boundary_ignore, use_for_loss=False),
        'lpips': LPIPS(boundary_ignore=boundary_ignore).to(device)
    }

    scores = {metric: [] for metric in metrics_all.keys()}
    out_dir = f'{base_results_dir}/synthetic/'
    os.makedirs(out_dir, exist_ok=True)

    for idx in tqdm.tqdm(range(len(dataset))):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']
        burst = burst.to(device).unsqueeze(0)
        gt = gt.to(device)

        with torch.no_grad():
            net_pred = net(burst)

        # For saving data
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
        cv2.imwrite(f'{out_dir}/{burst_name}.png', net_pred_np)
        # Post process image
        image = process_fn.process(net_pred.squeeze(0).cpu(), meta_info)
        cv2.imwrite(f'{out_dir}/{burst_name}_p.png', image)

        # Evaluation
        net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
        net_pred = net_pred_int.float() / (2 ** 14)

        with torch.no_grad():

            for metric, loss_fn in metrics_all.items():
                metric_value = loss_fn(net_pred, gt.unsqueeze(0)).cpu().item()
                scores[metric].append(metric_value)

    scores_all_mean = {"default": {metric: sum(values) / len(values) for metric, values in scores.items()}}

    report_text = generate_formatted_report(scores_all_mean)
    print(report_text)
