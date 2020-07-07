
import pathlib, os
from multiprocessing import Pool

import lib
from lib.utils import files
from lib.data import knee_cartilage

curr_path = pathlib.Path(__file__).parent.absolute()


def _check_paths(paths):
    for p in paths:
        assert os.path.isdir(p), f"{p}"

def _check_images(images):
    for im in images:
        assert files.is_image(im, checkfile=True), f"{im}"


def test_knee_cartilage():
    """
    Ensure each path, image, and mask are valid (both in loc and cardinality).
    """
    ds_path = os.path.join(curr_path, '..', '..', 'datasets', 'knee_cartilage')
    df = knee_cartilage.get_df(ds_path)
    
    paths = list(df['path'])
    images = list(df['images'])
    masks = list(df['masks'])
    with Pool() as p:
        p.map(_check_images, images + masks)
        p.map(_check_paths, [paths])

