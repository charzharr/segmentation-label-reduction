
import os, sys
import pathlib
from PIL import Image
import numpy as np

import torch

import lib
from lib.data import transforms, knee_cartilage


CURR_PATH = pathlib.Path(__file__).parent
NUM_WORKERS = 0


def get_data(cfg):
    ret = {}
    ret['df'] = df = knee_cartilage.get_df(os.path.join(
        CURR_PATH.parent.parent.parent.absolute(), 'datasets', 'knee_cartilage'
    ))
    
    ret['train_dataset'] = Knee2D(
        df[df['subsetname'] == 'train'],
        transforms.GeneralTransform(cfg['train']['transforms'])
    )
    ret['train_loader'] = torch.utils.data.DataLoader(
        ret['train_dataset'],
        batch_size=cfg['train']['batch_size'],
        shuffle=cfg['train']['shuffle'],
        num_workers=NUM_WORKERS, pin_memory=False  # non_block not useful here
    )

    ret['test_dataset'] = Knee2D(
        df[df['subsetname'] == 'test'],
        transforms.GeneralTransform(cfg['test']['transforms'])
    )
    ret['test_loader'] = torch.utils.data.DataLoader(
        ret['test_dataset'],
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False  # non_block not useful here
    )

    return ret


### ======================================================================== ###
### * ### * ### * ### *        Torch Data Handlers       * ### * ### * ### * ###
### ======================================================================== ###

class Knee2D(torch.utils.data.Dataset):
    
    def __init__(self, df, transforms):
        self.df = df
        self.T = transforms
        
        self.images = []
        [self.images.extend(s['images']) for _, s in df.iterrows()]
        self.masks = []
        [self.masks.extend(s['masks']) for _, s in df.iterrows()]
        assert len(self.images) == len(self.masks)
        for im, mask in zip(self.images, self.masks):
            assert im.split('/')[-2:] == mask.split('/')[-2:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        impath, maskpath = self.images[idx], self.masks[idx]
        X = self.T.transform(Image.open(impath).convert('L'), shake=True)
        
        mask = np.array(Image.open(maskpath).convert('L'))
        enc_mask = np.zeros((mask.shape[0], mask.shape[1], 5)).astype(np.uint8)
        for chan, val in enumerate((0, 50, 100, 150, 200)):
            enc_mask[:, :, chan] = np.equal(mask, val)*255
        Y = torch.cat([self.T.transform(Image.fromarray(enc_mask[:,:,c], 
                mode='L'), label=True) for c in range(5)])
        # print(Y.shape)
        # import matplotlib.pyplot as plt
        # for i in range(5):
        #     print(torch.max(Y[i,:,:]), torch.min(Y[i,:,:]))
        #     plt.imshow(Y[i,:,:])
        #     plt.show()
        return X, Y