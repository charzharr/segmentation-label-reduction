
from .baseline2d import emain as baseline2d
from .baseline3d import emain as baseline3d

EXPERIMENTS = {
    'baseline2d': baseline2d,
    'baseline3d': baseline3d,
}

def get_module(exp_name):
    if exp_name in EXPERIMENTS:
        return EXPERIMENTS[exp_name]
    else:
        for exp in EXPERIMENTS:
            if exp in exp_name:
                return EXPERIMENTS[exp]
