
from .unet2d import UNet 

def get_model(cfg, pretrained=False):
    ds_name = cfg['data']['name']
    bilinear = cfg['model']['unet2d']['bilinear']

    out_channels = len(cfg['data'][ds_name]['classnames'])
    in_channels = 0
    for entry in cfg['train']['transforms']:
        if entry[0] == 'normmeanstd':
            in_channels = len(entry[1][0])
    assert in_channels > 0, f"Could not get input dim from normmeanstd transform"
    
    model = UNet(in_channels, out_channels, bilinear=bilinear)
    return model