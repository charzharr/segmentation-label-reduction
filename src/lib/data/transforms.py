""" transforms.py (By: Charley Zhang, July 2020)
Define transform objects that allow for deterministic transforms.

Config Dict Specification:

"""

import random
import numbers
import collections
from PIL import Image, ImageFilter
import numpy as np

import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF

from lib.utils import images


class GeneralTransform:
    """ Transform object capable of handling images, and masks.
    Makes some assumptions for simplicity:
        1 - .transform() takes PILs and .reverse() takes tensors
        2 - same spatial and img_type transforms are done for X and Y
    """
    SUPPORTED_TRANSFORMS = set(['resize', 'togray', 'hflip', 'vflip', 
        'bcsh.jitter', 'gamma', 'totensor', 'topil', 'normmeanstd', 
        'gaussianblur', 'crop', 'rtss.affine'])
    LABEL_TRANSFORMS = set(['resize', 'hflip', 'vflip', 'totensor', 'topil', 
        'crop', 'rtss.affine'])
    INTERPOLATIONS = { 0: Image.NEAREST, 2: Image.BILINEAR, 3: Image.BICUBIC }

    def __init__(self, transform_list):
        assert isinstance(transform_list, collections.Sequence)
        self.transforms_cfg = transform_list
        self.transforms_det = None  # deterministic transforms to be populated

    def transform(self, im, label=False, shake=False, token=False):
        """ Transform an image. 
        Parameters
            img - PIL object to transform
            shake - if True, shakes up random values for transforms
            token - used in reverse() to reverse the transformations
        Returns
            [transformed_img, reverse_token]
        """
        assert isPIL(im), f"Got type {type(im)}."
        if shake or not self.transforms_det:
            self._shake()
        
        tokens = []
        for tname, settings in self.transforms_det:
            if label and tname not in GeneralTransform.LABEL_TRANSFORMS:
                continue
            im, tok = self._transform_im(
                im, tname, settings, reverse=False, label=label
            )
            if tok:
                tokens.append(tok)
        if token:
            return im, tokens
        return im

    def reverse(self, im, tokens, only_unnorm=False):
        # assert isinstance(im, torch.Tensor), f"Got type {type(im)}."
        if only_unnorm:
            t_names = list(zip(*tokens))[0]
            if 'normmeanstd' in t_names:
                settings = tokens[t_names.index('normmeanstd')][1]
                im, _ = self._transform_im(
                    im, 'normmeanstd', settings, reverse=True)
            if 'topil' in t_names:
                im, _ = self._transform_im(im, 'topil', True, reverse=True)
            return im
        for tname, settings in reversed(tokens):
            im = self._transform_im(im, tname, settings, reverse=True)
        return im

    def _transform_im(self, im, tname, settings, reverse=False, label=False):
        imsize = images.get_dimensions(im)[-2:]
        if tname == 'resize':
            out_size = settings[:2]
            it = 0 if label else 2
            return resize(im, out_size, interpolation=it), ['resize', [*imsize]]
        elif tname == 'crop':  # settings = h, w, ty, tx (h & w: ratio or abs)
            if 0 <= sum(settings) <= 2:
                h = int(min(imsize[0] * settings[0], imsize[0]))
                w = int(min(imsize[1] * settings[1], imsize[1]))
            else:
                h = int(min(settings[0], imsize[0]))
                w = int(min(settings[1], imsize[1]))
            ty = int(imsize[0] * settings[2])
            tx = int(imsize[1] * settings[3])
            assert 0 <= ty + h <= imsize[0] and 0 <= tx + w <= imsize[1]
            return crop(im, ty, tx, h, w), None
        elif tname == 'hflip':
            return hflip(im), ['hflip', True]
        elif tname == 'vflip':
            return vflip(im), ['vflip', True]
        elif tname == 'gamma':
            assert 0 <= settings, f"Invalid gamma settings: {settings}"
            return gamma(im, settings), ['gamma', 1/settings]
        elif tname == 'gaussianblur':
            return gaussian_blur(im, settings), None
        elif tname == 'bcsh.jitter':
            assert len(settings) == 4, f"Invalid jitter settings: {settings}"
            im = brightness(im, settings[0])
            im = contrast(im, settings[1])
            im = saturation(im, settings[2])
            im = hue(im, settings[3])
            rev = [1/v for v in settings[:3]] + [-settings[3]]
            return im, ['bcsh.jitter', rev]
        elif tname == 'rtss.affine':
            angle, translate, scale, shear = settings
            assert -1 <= translate[0] <= 1 and -1 <= translate[1] <= 1
            translate = int(translate[0]*imsize[0]), int(translate[1] * imsize[1])
            rev = [-angle, [-t for t in translate], 1/scale, [-s for s in shear]]
            return affine(im, int(angle), translate, scale, shear), \
                ['rtss.affine', rev]
        elif tname == 'togray':
            return togray(im, out_channels=1), None
        elif tname == 'totensor':
            return totensor(im), ['topil', True]
        elif tname == 'topil':
            return topil(im), ['totensor', True]
        elif tname == 'normmeanstd':
            assert len(settings) == 2, f"means, stds not lists: {settings}"
            im = normalize(im, *settings, reverse=reverse)
            return im, ['normmeanstd', settings]
        else:
            raise ValueError(f"Transform name '{tname}' is not support.")

    def _shake(self):
        """ Serves 2 main functions:
        (1) Sets self.transforms_det w/new random transforms based on cfg probs
        (2) Weeds out non-factors so that all entries are executed
        """
        transforms_det = []
        for tname, settings in self.transforms_cfg:
            assert tname in GeneralTransform.SUPPORTED_TRANSFORMS
            if not settings:
                continue
            if tname in ('togray', 'totensor', 'topil'):
                transforms_det.append([tname, settings])
            elif tname == 'resize':
                if isinstance(settings, numbers.Number):
                    transforms_det.append([tname, [settings, settings]])
                else:
                    assert len(settings) == 2, f"Given: {tname}, {settings}"
                    transforms_det.append([tname, settings])
            elif tname == 'crop':
                h_w_ty_tx = self._get_cropvals(settings)
                transforms_det.append([tname, h_w_ty_tx])
            elif tname == 'normmeanstd':
                assert len(settings) == 2, f"Given: {tname}, {settings}"
                transforms_det.append([tname, settings])
            elif tname == 'hflip' or tname == 'vflip':
                assert 0 <= settings <= 1.
                if random.uniform(0., 1.) <= settings:
                    transforms_det.append([tname, True])                
            elif tname == 'gamma':
                g = _uniform_sample(settings, center=1, bounds=(0,float('inf')))
                transforms_det.append([tname, g])
            elif tname == 'gaussianblur':
                if isinstance(settings, numbers.Number):
                    settings = (settings, settings)
                assert len(settings) == 2 and settings[0] <= settings[1]
                rad = round(_uniform_sample(settings, bounds=(0,float('inf'))))
                transforms_det.append([tname, rad])
            elif tname == 'bcsh.jitter':
                assert len(settings) == 4, f"Given: {tname}, {settings}"
                jittervals = self._get_jittervals(*settings)
                transforms_det.append([tname, jittervals])
            elif tname == 'rtss.affine':
                affinevals = self._get_affinevals(*settings)
                transforms_det.append([tname, affinevals])   
        self.transforms_det = transforms_det

    def _get_cropvals(self, settings):
        if isinstance(settings, numbers.Number):
            settings = [settings, settings]  # either (ratioh, ratiow) or (h, w)
        assert len(settings) == 2, f"Settings ({settings}) is not len 2."

        hwratios = []
        for v in settings:
            if isinstance(v, numbers.Number):
                assert 0 < v <= 1 and 0 < v <= 1
                hwratios.append(v)
            elif isinstance(v, collections.Sequence):
                assert len(v) == 2, f"Invalid crop h, w settings: {settings}."
                hwratios.append(_uniform_sample(v))
            else:
                raise ValueError(f"Gave invalid settings {settings} for crop.")
        ty = random.uniform(0, 1. - hwratios[0])
        tx = random.uniform(0, 1. - hwratios[1])
        assert ty + hwratios[0] <= 1 and tx + hwratios[1] <= 1
        return hwratios[0], hwratios[1], ty, tx

    def _get_jittervals(self, brightness, contrast, saturation, hue):
        jittervals = []
        for v in (brightness, contrast, saturation):
            jittervals.append(_uniform_sample(
                v, center=1, bounds=(.001, float('inf')))
            )
        jittervals.append(_uniform_sample(hue, center=0, bounds=(-0.5, 0.5)))
        return jittervals

    def _get_affinevals(self, rot, translate, scale, shear):
        """
        Parameters
            rot (num or seq) - num >= 0
            translate (num or seq) - 
        """
        if isinstance(rot, numbers.Number):
            assert rot >= 0
            rot = (-rot, rot)
        assert len(rot) == 2
        degree = _uniform_sample(rot, bounds=(-180, 180))

        translations = []
        if isinstance(translate, numbers.Number):
            translate = (translate, translate)  # translate for x, y
        for trans in translate:
            if isinstance(trans, numbers.Number):
                trans = (-trans, trans)
            assert 2 == sum([-1 <= t <= 1 for t in trans])
            translations.append(_uniform_sample(trans, bounds=(-1,1)))
        
        if isinstance(scale, numbers.Number):
            if not scale:
                scale = 1
            assert scale > 0
        else:
            assert len(scale) == 2
            scale = _uniform_sample(scale, bounds=(0,float('inf')))
        
        shears = []
        if isinstance(shear, numbers.Number):
            shear = (shear, shear)
        for s in shear:
            if isinstance(s, numbers.Number):
                s = (-s, s)
            assert len(s) == 2
            shears.append(_uniform_sample(s, bounds=(-180,180)))

        return [degree, translations, scale, shears]


### Tranform Helpers

def _uniform_sample(val, center=1, bounds=(0, float('inf'))):
    """ Sample from uniform distributions with parameters given.
    Parameters
        val (list or number) - list indicates range, val represents offset
        center (number) - only used if val is number
        bounds (list: 2 numbers) - indicates min & max bound to sample from
    """
    if not val:  # is None, 0, or an empty sequence
        return center
    elif isinstance(val, collections.Sequence):
        assert len(val) == 2, f"Value seq not valid ({v})."
        assert val[0] <= val[1], f"Req: v0 <= v1. Got: ({val[0]}, {val[1]})."
        return random.uniform(max(bounds[0], val[0]), min(bounds[1], val[1]))
    else:
        assert isinstance(val, numbers.Number), f"Val ({val}) has invalid type."
        return random.uniform(
            max(bounds[0], center - val), min(bounds[1], center + val)
        )  


### ======================================================================== ###
### * ### * ### * ### *   Main Raster ImageTransforms    * ### * ### * ### * ###
### ======================================================================== ###

def isPIL(obj):
    return True if 'PIL' in str(obj.__class__) else False

def resize(pil, out_size, interpolation=2):
    assert isPIL(pil), f"Got type {type(pil)}."
    return TFF.resize(pil, out_size, interpolation=interpolation)

def crop(pil, tl_y, tl_x, h, w):
    assert isPIL(pil), f"Got type {type(pil)}."
    return TFF.crop(pil, tl_y, tl_x, h, w)

def vflip(im):
    assert isPIL(im) or isinstance(im, torch.Tensor), f"Got type {type(im)}."
    return TFF.vflip(im)  # returns PIL

def hflip(im):
    assert isPIL(im) or isinstance(im, torch.Tensor), f"Got type {type(im)}."
    return TFF.hflip(im)  # returns PIL

def affine(pil, angle, translate, scale, shear):
    assert isPIL(pil), f"Got type {type(pil)}."
    return TFF.affine(pil, angle, translate, scale, shear)

def rotate(pil, angle):
    assert isPIL(pil), f"Got type {type(pil)}."
    return TFF.rotate(pil, angle)

def affine(im, angle, translate, scale, shear):
    assert isPIL(im), f"Got type {type(im)}."
    assert -180 <= angle <= 180, f"Invalid angle: {angle}"
    return TFF.affine(im, angle, translate, scale, shear)

def totensor(im):
    assert isPIL(im) or isinstance(np.ndarray), f"Got type {type(im)}."
    return TFF.to_tensor(im)

def topil(im):
    if isinstance(im, torch.Tensor):
        if im.shape[0] == 1 or im.shape[0] == 3:
            # im = im.permute(1,2,0)  # im is still float32
            pass
    elif isinstance(im, np.ndarray):
        if im.shape[0] == 1 or im.shape[0] == 3:
            im = np.moveaxis(im, 0, -1)
        if np.max(im) > 1:
            im = im.astype(np.uint8)
        else:
            print(f"WARNING: np image is full of floats.")
    else:
        raise ValueError(f"Expects np.array or tensor, got {type(im)}.")
    # mode = 'gray' if im.shape[-1] == 1 else 'RGB'
    pil = TFF.to_pil_image(im, mode=None)
    return pil

def togray(pil, out_channels):
    assert isPIL(pil), f"Expected PIL, got type {type(pil)}."
    return TFF.to_grayscale(pil, num_output_channels=out_channels)

def normalize(tens, mean, std, reverse=False):
    assert isinstance(tens, torch.Tensor), f"Got type {type(tens)}."
    if reverse:
        meant = torch.tensor(mean)
        stdt = torch.tensor(std)
        C = tens.shape[0]
        assert meant.shape[0] == stdt.shape[0] == C
        return tens * stdt.view(C, 1, 1) + meant.view(C, 1, 1)
    return TFF.normalize(tens, mean, std)

def gaussian_blur(pil, radius):
    assert isPIL(pil), f"Expected PIL, got type {type(pil)}."
    return pil.filter(ImageFilter.GaussianBlur(radius=radius))

## Color Jitter
def brightness(im, factor):
    if factor == 1:
        return im
    assert isPIL(im) or isinstance(torch.Tensor), f"Got type {type(im)}."
    return TFF.adjust_brightness(im, factor)

def contrast(im, factor):
    if factor == 1:
        return im
    assert isPIL(im) or isinstance(torch.Tensor), f"Got type {type(im)}."
    return TFF.adjust_contrast(im, factor)

def hue(im, factor):
    if factor == 0:
        return im
    assert isPIL(im) or isinstance(torch.Tensor), f"Got type {type(im)}."
    return TFF.adjust_hue(im, factor)

def saturation(im, factor):
    if factor == 1:
        return im
    assert isPIL(im) or isinstance(torch.Tensor), f"Got type {type(im)}."
    return TFF.adjust_saturation(im, factor)

def gamma(pil, gamma, gain=1):
    if gamma == 1 and gain == 1:
        return pil
    assert isPIL(pil), f"Got type {type(im)}."
    return TFF.adjust_gamma(pil, gamma, gain=gain)



### ======================================================================== ###
### * ### * ### * ### *    Main Coordinate Transforms    * ### * ### * ### * ###
### ======================================================================== ###

class CoordinateTransform(GeneralTransform):
    """ Transform object capable of handling images, masks, and coordinates.
    Makes some assumptions for simplicity:
        1 - .transform() takes PILs and .reverse() takes tensors
        2 - same spatial and img_type transforms are done for X and Y
    """
    COORD_TRANSFORMS = set(['resize', 'hflip', 'vflip', 'crop', 'rtss.affine'])

    def __init__(self, transform_list):
        super(CoordinateTransform, self).__init__(transform_list)

    def transform_coord(self, im, shake=False, token=False):
        pass


def resize_coord():
    pass

def crop_coord():
    pass

def flip_coord():
    pass

def affine_coord():
    pass
