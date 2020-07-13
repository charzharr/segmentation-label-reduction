""" esetup.py (By: Charley Zhang, July 2020)
Does the heavy lifting for setting up the experiment.
  - Gathers the necessary resources, modules, and utilities.
  - Configures all components used in training
  - Initializes stat trackers
"""


import sys, os
import torch

import lib
from lib.utils import schedulers, statistics
from lib.modules import losses2d, unet2d

from . import edata

MODELS = {
    'unet2d': unet2d,
    # 'unet3d': unet3d,
}


def setup(cfg, checkpoint):
    """
    1. Verify configuration
    2. Load major training components
        - Data  - Model  - Optimizer  - Criterion  - Recording
    """

    # Print Settings
    print(f"[Experiment Settings (@esetup.py)]")
    print(f" > Prepping train config..")
    print(f"\t - experiment:  {cfg['experiment']['project']} - "
            f"{cfg['experiment']['name']}, id({cfg['experiment']['id']})")
    print(f"\t - batch_size {cfg['train']['batch_size']}, "
          f"\t - start epoch: {cfg['train']['start_epoch']}/"
            f"{cfg['train']['epochs']},")
    print(f"\t - Optimizer ({cfg['train']['optimizer']['name']}): "
          f"\t - lr {cfg['train']['optimizer']['lr']}, "
          f"\t - wt_decay {cfg['train']['optimizer']['wt_decay']}, "
          f"\t - mom {cfg['train']['optimizer']['momentum']}, ")
    print(f"\t - Scheduler ({cfg['train']['scheduler']['name']}): "
          f"\t - factor {cfg['train']['scheduler']['factor']}, ")

    # Flags
    use_wandb = cfg if not cfg['experiment']['debug']['mode'] and \
                cfg['experiment']['debug']['wandb'] else None

    # Load Model Components
    data = edata.get_data(cfg)
    device = torch.device(cfg['experiment']['device'])
    criterion = _get_criterion(cfg)
    model = _get_model(cfg)

    if checkpoint:
        resume_dict = torch.load(checkpoint)
        cfg['train']['start_epoch'] = resume_dict['epoch']
        
        state_dict = resume_dict['state_dict']
        print(' > ' + str(model.load_state_dict(state_dict, strict=False)))
        
        optimizer = resume_dict['optimizer']
        
        if 'scheduler' in resume_dict:
            scheduler = resume_dict['scheduler']
        else:
            scheduler = _get_scheduler(cfg, optimizer)

        if 'tracker' in resume_dict:
            tracker = resume_dict['tracker']
            tracker.use_wandb = use_wandb
            # TODO: resuming log
        else:
            tracker = utils.statistics.ExperimentTracker(wandb=use_wandb)
            
    else:
        optimizer = _get_optimizer(cfg, model.parameters())
        scheduler = _get_scheduler(cfg, optimizer)
        tracker = statistics.ExperimentTracker(wandb=use_wandb)

    return {
        'device': device,
        'data': data,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'tracker': tracker
    }


### ---- ### ---- \\    Helpers     // ---- ### ---- ###


def _get_model(cfg):
    name = cfg['model']['name']
    if name.lower() in MODELS:
        print(f"  > Fetching {name} model..", end='')
        model = MODELS[name].get_model(cfg, pretrained=False)
    else:
        raise ValueError(f"  > Model({name}) not found..")
    return model


def _get_scheduler(cfg, optimizer):
    sched = cfg['train']['scheduler']['name']
    t = cfg['train']['start_epoch']
    T = cfg['train']['epochs']
    factor = cfg['train']['scheduler']['factor']
    
    if 'plateau' in sched:
        scheduler = schedulers.ReduceOnPlateau(
            optimizer,
            factor=factor,
            patience=cfg['train']['scheduler']['plateau']['patience'],
            lowerbetter=True
        )
    elif 'step' in sched:
        scheduler = schedulers.StepDecay(
            optimizer,
            factor=factor,
            T=T,
            steps=cfg['train']['scheduler']['step']['steps']
        )
    elif 'cos' in sched:
        scheduler = schedulers.CosineDecay(
            optimizer,
            T=T,
            t=t
        )
    else:
        scheduler = schedulers.Uniform(optimizer)
    
    return scheduler


def _get_optimizer(cfg, params):
    opt = cfg['train']['optimizer']['name']
    lr = cfg['train']['optimizer']['lr']
    mom = cfg['train']['optimizer']['momentum']
    wdecay = cfg['train']['optimizer']['wt_decay']

    if 'adam' in opt:
        optimizer = torch.optim.Adam(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=cfg['train']['optimizer']['adam']['betas']
        )
    elif 'nesterov' in opt:
        optimizer = torch.optim.SGD(params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay,
            nesterov=True
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay
        )

    return optimizer


def _get_criterion(cfg):
    
    critname = cfg['criterion']['name']
    
    if 'bce_logit' in critname:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif 'pixelwise_bce_2d':
        criterion = losses2d.PixelWiseBCE()
    elif 'bce' in critname:
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f"Criterion {critname} is not supported.")

    return criterion


