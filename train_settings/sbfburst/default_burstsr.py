import torch
import torch.optim as optim

import actors.sbfb_actors as sbfb_actors
import dataset as datasets
from admin.environment import env_settings
from data import processing, sampler, DataLoader
from models.alignment.pwcnet import PWCNet
from models.loss.image_quality_v2 import PSNR
from models.loss.mixg_loss import PixelWiseMGLError
from models.sbfburst.sbfburst_raw import SBFBurstRAW
from trainers import SimpleTrainer


def run(settings):
    settings.batch_size = 1
    settings.num_workers = 2
    settings.print_interval = 10

    settings.crop_sz = 56
    settings.burst_sz = 14

    data_processing_train = processing.BurstSRProcessing(transform=None, random_flip=True,
                                                         substract_black_level=True,
                                                         crop_sz=settings.crop_sz)
    burstsr_train = datasets.BurstSRDataset(split='train')

    data_processing_val = processing.BurstSRProcessing(transform=None,
                                                       substract_black_level=True, crop_sz=settings.crop_sz)
    burstsr_val = datasets.BurstSRDataset(split='val')

    # Train sampler and loader
    dataset_train = sampler.RandomBurst([burstsr_train], [1], burst_size=settings.burst_sz,
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)

    # Train sampler and loader
    dataset_val = sampler.IndexedBurst(burstsr_val, burst_size=settings.burst_sz, processing=data_processing_val)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)

    net = SBFBurstRAW()
    # net = torch.compile(net)

    objective = {
        'rgb': PixelWiseMGLError(boundary_ignore=None),
        'psnr': PSNR(boundary_ignore=40)
    }

    loss_weight = {
        'rgb': 10.0,
    }

    pwcnet = PWCNet(load_pretrained=True,
                    weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))
    actor = sbfb_actors.SBFBRealWorldActor(net=net, objective=objective, loss_weight=loss_weight, alignment_net=pwcnet)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 5e-5}])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    net.load_checkpoint('./pretrained_networks/pretrained_synthetic.pth.tar')

    trainer.train(80, load_latest=True, fail_safe=False)
