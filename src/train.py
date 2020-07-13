""" Sets up and calls the appropriate experiment module.
1. Setup
   a. Get YAML config file + validate
"""
import sys, os
import math, random
import numpy as np

import wandb
import click
import torch

import configs, experiments

USER_CHOICES = ("charzhar", "yzhang46")


@click.command()
@click.option("--user", type=click.Choice(USER_CHOICES, case_sensitive=True))
@click.option("--gpu", default=0)
@click.option("--config", type=click.Path(exists=True))
@click.option("--checkpoint", type=str)
def run_cli(user, gpu, config, checkpoint=None):
    if checkpoint:
        assert os.path.isfile(checkpoint), \
        f"Checkpoint doesn't exist ({checkpoint})"

    """ Get config + env setup. """
    cfg = configs.get_config(config)

    # experiment setup
    _set_seed(cfg['experiment']['seed'])

    if user == 'yzhang46' and gpu >= 0:
        device = f'cuda:{int(gpu)}' if torch.cuda.is_available() else 'cpu'
    else:
        device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg['experiment']['device'] = device
    print(f" > Using device: {device}.")
    
    experiment = experiments.get_module(cfg['experiment']['name'])
    experiment.run(cfg, checkpoint=checkpoint)


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


if __name__ == '__main__':
    run_cli()