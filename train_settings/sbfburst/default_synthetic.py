import torch
import torch.optim as optim

import actors.sbfb_actors as sbfb_actors
import data.transforms as tfm
import dataset as datasets
from data import processing, sampler, DataLoader
from models.loss.image_quality_v2 import PSNR
from models.loss.mixg_loss import PixelWiseMGLError
from models.sbfburst.sbfburst_raw import SBFBurstRAW
from models.utils import put_requires_grad
from trainers.simple_trainer_autocast import SimpleTrainerAutocast


def run(settings):
    settings.batch_size = 1
    settings.num_workers = 2
    settings.print_interval = 10

    settings.crop_sz = (384, 384)
    settings.burst_sz = 14
    settings.downsample_factor = 4

    settings.burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24}
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True,
                                        'add_noise': True}

    zurich_raw2rgb_train = datasets.ZurichRAW2RGB(split='train')
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='test')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())

    data_processing_train = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                                settings.downsample_factor,
                                                                burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params)
    data_processing_val = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                              settings.downsample_factor,
                                                              burst_transformation_params=settings.burst_transformation_params,
                                                              transform=transform_val,
                                                              image_processing_params=settings.image_processing_params)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.RandomImage([zurich_raw2rgb_val], [1],
                                      samples_per_epoch=settings.batch_size * 200, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size, pin_memory=True)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5, pin_memory=True)

    net = SBFBurstRAW()
    # net = torch.compile(net)

    objective = {'rgb': PixelWiseMGLError(boundary_ignore=None), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actor = sbfb_actors.SBFBSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.AdamW([{'params': actor.net.parameters(), 'lr': 2e-4}])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 200, 280, 350, 410, 460], gamma=0.5)
    trainer = SimpleTrainerAutocast(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    put_requires_grad(net, True)
    put_requires_grad(net.spynet, False)

    trainer.train(130, load_latest=True, fail_safe=True)

    put_requires_grad(net, True)
    put_requires_grad(net.spynet, True)

    trainer.train(500, load_latest=True, fail_safe=True)
