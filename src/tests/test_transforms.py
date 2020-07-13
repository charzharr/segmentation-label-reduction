
import pathlib, os, sys
from multiprocessing import Pool
import PIL
import matplotlib.pyplot as plt

import lib
from lib.data import transforms

curr_path = pathlib.Path(__file__).parent.absolute()

def test_frame_transform():
    def test_cfg(cfg, im):
        T = transforms.FrameTransform(cfg)
        tim, tok = T.transform(im, token=True)
        uim = T.reverse(tim, tok, only_unnorm=True)
        print(tim)
        sys.exit(0)
        
        # assert np.sum(np.array(im) - ) < 10**-4 

    im = PIL.Image.open(os.path.join(curr_path, 'sample.png'))
    plt.imshow(im)

    cfg = [ ['hflip', 1], ['colorjitter', [0.1, 0.1, 0.1, 0.1]],
            ['totensor', True], 
            ['normmeanstd', [[0.15001505461961784], [0.10547640998002673]]] ]
    test_cfg(cfg, im)

if __name__ == '__main__':
    test_frame_transform()
    
    