from pathlib import Path

import colour_demosaicing
import cv2
import pandas as pd
import torch
from tqdm import tqdm

from admin.environment import env_settings
from data.postprocessing_functions import process_samsung
from data.raw_image_processing import flatten_raw_image
from dataset.raw56_dataset import get_newbsr_val_set
from models.alignment.pwcnet import PWCNet
from models.loss.aligned_metrics import AlignedLPIPS, AlignedPSNR, AlignedSSIM
from models.sbfburst.sbfburst_raw import SBFBurstRAW

FRAME_COUNTS = [14, 28, 42, 56]
CHECKPOINTS = ['pretrained_burstsr', 'pretrained_raw56_14frames', 'pretrained_raw56_allframes']
COLUMNS = ['nf', 'chk', 'scene', 'idx', 'psnr', 'ssim', 'lpips']


def evaluate_configuration(nf, chk, dataset, base_dir, metric_functions, device):

    out_dir = base_dir / f'{nf}_{chk}'
    out_dir.mkdir(parents=True, exist_ok=True)

    net = SBFBurstRAW()
    net.load_checkpoint(f'./pretrained_networks/{chk}.pth.tar')
    net.to(device).eval()

    rows = []
    for idx in tqdm(range(len(dataset)), desc=f'{nf} frames | {chk}'):
        data = dataset[idx]
        scene = str(data['burst_name'])
        meta_info = data['meta_info_burst']
        gt = data['frame_gt'].unsqueeze(0).to(device)
        burst = data['burst'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(burst)

            # Save PNGs using the original Samsung post-processing (before quantization).
            image = flatten_raw_image(burst[0][0].cpu())
            image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(image, pattern='RGGB')
            image = process_samsung(torch.from_numpy(image.transpose(2, 0, 1)), meta_info)
            cv2.imwrite(str(out_dir / f'{idx}_i.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            image = process_samsung(pred.squeeze(0).cpu(), meta_info)
            cv2.imwrite(str(out_dir / f'{idx}_p.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            image = process_samsung(gt.squeeze(0).cpu(), meta_info)
            cv2.imwrite(str(out_dir / f'{idx}_gt.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Preserve the exact original 14-bit quantization.
            pred = (pred.clamp(0.0, 1.0) * (2 ** 14)).short().float() / (2 ** 14)
            row = {'nf': nf, 'chk': chk, 'scene': scene, 'idx': idx}
            for metric, loss_fn in metric_functions.items():
                row[metric] = float(loss_fn(pred, gt, burst).detach().cpu().item())

        rows.append(row)

    result = pd.DataFrame(rows, columns=COLUMNS)

    return result


def main():
    settings = env_settings()
    base_dir = Path(settings.save_predictions_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    scores_path = base_dir / 'scores_per_scene.csv'

    scores_path.unlink(missing_ok=True)

    device = 'cuda'
    pwcnet = PWCNet(
        load_pretrained=True,
        weights_path=f'{settings.pretrained_nets_dir}/pwcnet-network-default.pth'
    ).to(device).eval()
    metric_functions = {
        'psnr': AlignedPSNR(pwcnet),
        'ssim': AlignedSSIM(pwcnet),
        'lpips': AlignedLPIPS(pwcnet).to(device),
    }

    all_rows = []
    for nf in FRAME_COUNTS:
        dataset = get_newbsr_val_set(nf)
        print(f'\n{nf} frames: {len(dataset)} test scenes')

        for chk in CHECKPOINTS:
            result = evaluate_configuration(
                nf, chk, dataset, base_dir, metric_functions, device
            )
            all_rows.extend(result.to_dict('records'))

            tmp_path = base_dir / 'scores_per_scene.csv.tmp'
            pd.DataFrame(all_rows, columns=COLUMNS).to_csv(tmp_path, index=False)
            tmp_path.replace(scores_path)
            print(f'[SAVED] {scores_path}: {len(all_rows)} rows')

    print(f'\nCompleted: {scores_path}')


if __name__ == '__main__':
    main()
