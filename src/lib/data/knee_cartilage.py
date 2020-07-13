
import sys, os
import pandas as pd
import pickle
import PIL
import torch


def encode_Y(mask_tensor):
    assert isinstance(mask_tensor, torch.Tensor)
    ret = torch.zeros((5, *mask_tensor.shape[-2:]))
    for channel, val in enumerate((0, 50, 100, 150, 200)):
        ret[channel,...] = mask_tensor.eq(val).float()
    return ret

def get_df(path, df_file='df.pd'):
    """ Returns the knee cartilage dataset as a df. """
    def _replace(it):
        if isinstance(it, str):
            return it.replace(base_path, path)
        new_it = []
        for i in it:
            new_it.append(i.replace(base_path, path))
        return new_it
    
    assert os.path.isdir(path)

    if os.path.isfile(os.path.join(path, df_file)):
        with open(os.path.join(path, df_file), 'rb') as f:
            df = pickle.load(f)
    else:
        raise ValueError(f"DF file not in: {os.path.join(path, df_file)}")

    # Fix paths
    old_path_s = df.iloc[0]['path'].split('/')

    if 'Train' in old_path_s:
        idx = old_path_s.index('Train')
        base_path = '/' + os.path.join(*old_path_s[:idx])
    else:
        idx = old_path_s.index('Test')
        base_path = '/' + os.path.join(*old_path_s[:idx])

    df['path'] = df['path'].apply(_replace)
    df['images'] = df['images'].apply(_replace)
    df['masks'] = df['masks'].apply(_replace)

    return df

        
    


    