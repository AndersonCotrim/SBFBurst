import actors.sbfb_actors as sbfb_actors
from admin.environment import env_settings
from data import processing, sampler, DataLoader
from dataset.raw56_dataset import Raw56Dataset
from models.alignment.pwcnet import PWCNet
from models.loss.image_quality_v2 import PSNR, PixelWiseError
from models.sbfburst.sbfburst_raw import SBFBurstRAW
from trainers import SimpleTrainer

import schedulefree

def run(settings):
    settings.batch_size = 2
    settings.num_workers = 2
    settings.print_interval = 10

    settings.crop_sz = 56
    settings.burst_sz = 14

    data_processing_train = processing.Raw56Processing(transform=None, random_flip=True,
                                                         substract_black_level=False,
                                                         crop_sz=settings.crop_sz)

    burstsr_train = Raw56Dataset(split='train', crop_sz=settings.crop_sz, burst_size=settings.burst_sz, random_sampling=True)

    # Train sampler and loader
    dataset_train = sampler.RandomBurst([burstsr_train], [1], burst_size=settings.burst_sz,
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)


    net = SBFBurstRAW()
    #net = torch.compile(net)

    objective = {
        'rgb': PixelWiseError(boundary_ignore=None),
        'psnr': PSNR(boundary_ignore=40)
    }

    loss_weight = {
        'rgb': 10.0,
    }

    pwcnet = PWCNet(load_pretrained=True,
                    weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))
    actor = sbfb_actors.SBFBRealWorldActor(net=net, objective=objective, loss_weight=loss_weight, alignment_net=pwcnet)

    optimizer = schedulefree.AdamWScheduleFree(
                actor.net.parameters(),
                lr= 5e-4,
                warmup_steps=50,
            )

    trainer = SimpleTrainer(actor, [loader_train], optimizer, settings, None)

    net.load_checkpoint('./pretrained_networks/pretrained_synthetic.pth.tar')

    trainer.train(160, load_latest=True)
