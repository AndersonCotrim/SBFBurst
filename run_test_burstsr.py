import os

import cv2
import numpy as np
import torch
import tqdm

from admin.environment import env_settings
from data.postprocessing_functions import process_canon
from dataset.burstsr_dataset import get_burstsr_val_set
from models.alignment.pwcnet import PWCNet
from models.loss.aligned_metrics import AlignedLPIPS, AlignedPSNR, AlignedSSIM
from models.sbfburst.sbfburst_raw import SBFBurstRAW
from utils.display_utils import generate_formatted_report

if __name__ == '__main__':
    base_results_dir = env_settings().save_predictions_path
    dataset = get_burstsr_val_set()

    net = SBFBurstRAW()
    net.load_checkpoint("./pretrained_networks/pretrained_burstsr.pth.tar")
    net.to('cuda').train(False)

    device = 'cuda'
    pwcnet = PWCNet(load_pretrained=True,
                    weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))

    pwcnet = pwcnet.cuda()
    pwcnet = pwcnet.eval()

    metrics_all = {
        'psnr': AlignedPSNR(pwcnet),
        'ssim': AlignedSSIM(pwcnet),
        'lpips': AlignedLPIPS(pwcnet).to(device)
    }

    scores = {metric: [] for metric in metrics_all.keys()}
    out_dir = f'{base_results_dir}/burstsr/'
    os.makedirs(out_dir, exist_ok=True)

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        gt = data['frame_gt'].unsqueeze(0)
        burst = data['burst'].unsqueeze(0)
        burst_name = data['burst_name']
        meta_info = data['meta_info_burst']

        burst = burst.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            net_pred = net(burst)

        # For saving data
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
        cv2.imwrite(f'{out_dir}/{burst_name}.png', net_pred_np)
        # Post process image
        image = process_canon(net_pred.squeeze(0).cpu(), meta_info)
        cv2.imwrite(f'{out_dir}/{burst_name}_p.png', image)

        # Evaluation
        net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
        net_pred = net_pred_int.float() / (2 ** 14)
        with torch.no_grad():
            for metric, loss_fn in metrics_all.items():
                metric_value = loss_fn(net_pred, gt, burst).cpu().item()
                scores[metric].append(metric_value)

    scores_all_mean = {"default": {metric: sum(values) / len(values) for metric, values in scores.items()}}

    report_text = generate_formatted_report(scores_all_mean)
    print(report_text)
