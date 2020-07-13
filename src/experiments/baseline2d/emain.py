"""
Customized for this experiment:
  Characteristics:
    - Single frame examples
    - Batches with certain ratio of inputs with labels
  Custom Functions
    - create_batches(examples, batch_size, has_label_ratio)
"""

import sys, os
import time
import pathlib
import collections
import pprint
import random, math
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch, torchvision
from torchvision import transforms
from torchsummary import summary

import lib
from lib import metrics2d, metrics3d
from lib.utils import statistics, images, devices, timers
from lib.data import knee_cartilage

from . import edata, esetup


WATCH = timers.StopWatch()
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'test_ep_dice': True,
}

def run(cfg, checkpoint=None):

    def it_metrics(batch_d, out_d, loss, meter, tracker, test=False):
        itmets = {'loss': float(loss)}
        itmets['dice'] = metrics2d.dice(out, Y)
        itmets['iou'] = metrics2d.IoU(out, Y)
        meter.update(itmets)
        
        pre = 'test_' if test else 'train_' 
        tracker.update({
            pre + 'it_loss': itmets['loss'],
            pre + 'it_dice': itmets['dice'],
            pre + 'it_iou': itmets['iou']
        })
        return itmets
    
    # Training Components
    debug = cfg['experiment']['debug']
    comps = esetup.setup(cfg, checkpoint)

    data = comps['data']
    trainloader = data['train_loader']
    testloader = data['test_loader']

    device = comps['device']
    model = comps['model'].to(device)
    criterion = comps['criterion'].to(device)
    optimizer = comps['optimizer']
    scheduler = comps['scheduler']
    tracker = comps['tracker']

    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']+1):
        print("\n======================")
        print(f"Starting Epoch {epoch} (lr: {scheduler.lr})")
        print("======================")
        WATCH.tic(name='epoch')

        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic(name='iter')
        for it, batch in enumerate(trainloader):
            
            X = batch[0].to(device)
            Y = batch[1].to(device)

            out = model(X)  # list of out_ds
    
            optimizer.zero_grad()
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            X, Y, out = X.detach(), Y.detach(), out.detach()

            itmets = it_metrics(Y, out, loss.item(), epmeter, tracker)
            _print_itmets(cfg, it+1, len(trainloader), itmets,
                WATCH.toc(name='iter', disp=False))
            
            if debug['break_train_iter']: break
            WATCH.tic(name='iter')

            sys.exit(0)

        # Test + Epoch Metrics
        if epoch % debug['test_every_n_epochs'] == 0:
            print(f"\nTesting..\n")
            with torch.no_grad():
                model.eval()
                testmeter = statistics.EpochMeters()
                WATCH.tic(name='iter')
                for it, batch_d in enumerate(testloader):
                    X = batch_d['X'].to(device)
                    Y = {k: v.to(device) for k, v in batch_d['Y'].items()}
                    loss, loss_ds, out_ds = forward(X, Y)

                    out_ds = [{k: v.detach() for k, v in o.items()} for o in out_ds]
                    itmets = it_metrics(batch_d, out_ds[-1], 
                        loss.item(), testmeter, tracker, test=True
                    )
                    _print_itmets(cfg, it+1, len(testloader), itmets, 
                        WATCH.toc(name='iter', disp=False))
                    if debug['break_test_iter']: break
                    WATCH.tic(name='iter')
            epmets = _epoch_mets( 
                cfg, tracker,
                epmeter.avg(no_avg=['TPs', 'FPs', 'FNs']), 
                testmeter.avg(no_avg=['TPs', 'FPs', 'FNs'])
            )
        else:
            epmets = _epoch_mets(
                cfg, tracker, 
                epmeter.avg(no_avg=['TPs', 'FPs', 'FNs'])
            )
        
        WATCH.toc(name='epoch')
        scheduler.step(epoch=epoch, value=0)

        if debug['save']:
            _save_model(epmets, model.state_dict(), criterion, optimizer, 
                tracker,cfg)



### ======================================================================== ###
### * ### * ### * ### *              Helpers             * ### * ### * ### * ###
### ======================================================================== ###


def _print_itmets(cfg, iter_num, iter_tot, it_mets, duration):
    dev = cfg['experiment']['device']
    mem = devices.get_gpu_memory_map()[int(dev[-1])]/1000 if 'cuda' in dev else -1
    print(
        f"\n    Iter {iter_num}/{iter_tot} ({duration:.1f} sec, {mem:.1f} GB) - "
        f"loss {float(it_mets['loss']):.3f}, "
        f"dice {float(it_mets['dice']):.3f}, "
        f"iou {float(it_mets['iou']):.3f}, "
    )

# dict_keys: loss, mAP, F1, F1s, TPs, FPs, FNs
def _epoch_mets(cfg, tracker, *dicts):
    classnames = cfg['data']['classnames']
    merged = {}
    for i, d in enumerate(dicts):
        pre = 'test_ep_' if i else 'train_ep_'
        class_sep = False if len(classnames) == 1 else True
        for k, v in d.items():  # add class confs, mAP, and loss
            if k in ['F1', 'F1s']: 
                continue
            if k in ['TPs', 'FPs', 'FNs'] and class_sep:
                for cidx in range(len(v)):
                    merged[pre + k[:-1] + f'_{cidx}'] = v[cidx]
            if k in ['mAP', 'loss']:
                merged[pre + k] = v
        totTP, totFP, totFN = sum(d['TPs']), sum(d['FPs']), sum(d['FNs'])
        merged[pre + 'TP'] = totTP
        merged[pre + 'FP'] = totFP
        merged[pre + 'FN'] = totFN
        merged[pre + 'recall'] = totTP / (totTP + totFN + 10**-5)
        merged[pre + 'precision'] = totTP / (totTP + totFP + 10**-5)
        merged[pre + 'F1'] = (2*totTP) / (2*totTP + totFP + totFN + 10**-5)
                    
    tracker.update(merged, wandb=True)
    
    print("\nEpoch Stats\n-----------")
    for k, v in merged.items():
        print(f"  {k}: {v}")
    return merged
    
    
def _save_model(epmets, state, crit, opt, tracker, cfg):
    print(f"Saving model ", end='')
    end = 'last'
    for met, max_gud in SAVE_BEST_METRICS.items():
        if met in epmets and \
        tracker.best(met, max_better=max_gud) == epmets[met]:
            end = f"best-{met.split('_')[-1]}"
            print(f"({end}: {epmets[met]:.2f}) ", end='')
            break
    filename = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_" + end
    print(f"-> {filename}")

    curr_path = pathlib.Path(__file__).parent.absolute()
    save_path = os.path.join(curr_path, filename + '.pth')
    torch.save({
        'state_dict': state,
        'criterion': crit,
        'optimizer': opt,
        'tracker': tracker,
        'config': cfg
        },
        save_path
    )

