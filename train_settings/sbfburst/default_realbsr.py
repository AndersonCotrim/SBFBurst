import torch
import torch.optim as optim

import actors.sbfb_actors as sbfb_actors
from data import sampler, DataLoader
from data.processing_rbsr import RBSRProcessing
from dataset.realbsr_dataset import RealBSRDataset
from models.loss.image_quality_v2 import PSNR
from models.loss.mixg_loss import PixelWiseMGLError
from models.sbfburst.sbfburst_rgb import SBFBurstRGB
from models.utils import put_requires_grad
from trainers.simple_trainer_autocast import SimpleTrainerAutocast


def run(settings):
    settings.batch_size = 4
    settings.num_workers = 2
    settings.print_interval = 10

    settings.crop_sz = None
    settings.burst_sz = 14

    data_processing_train = RBSRProcessing(transform=None, crop_sz=settings.crop_sz)

    train_dataset = RealBSRDataset(split='train', mode="RGB", aligned=False)

    train_dataset = sampler.RandomBurst([train_dataset], [1], burst_size=settings.burst_sz,
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)

    val_dataset = RealBSRDataset(split='val', mode="RGB", aligned=False)

    val_dataset = sampler.IndexedBurst(val_dataset, burst_size=settings.burst_sz)

    loader_train = DataLoader('train', train_dataset, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size, pin_memory=True)

    loader_val = DataLoader('val', val_dataset, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5, pin_memory=True)

    net = SBFBurstRGB()
    net = torch.compile(net)

    criterion = PixelWiseMGLError(boundary_ignore=None)

    objective = {'rgb': criterion, 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actor = sbfb_actors.SBFBRealBSRActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.AdamW([{'params': actor.net.parameters(), 'lr': 2e-4}])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.5)
    trainer = SimpleTrainerAutocast(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    put_requires_grad(net, True)
    put_requires_grad(net.spynet, False)

    trainer.train(105, load_latest=True, fail_safe=True)
