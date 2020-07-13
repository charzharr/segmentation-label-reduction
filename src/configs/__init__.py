
import os
import pathlib
import yaml

__all__ = ['get_config']


def get_config(filename):
    print(f" > Loading config ({filename})..", end='')
    
    if os.path.isfile(filename):
        cfg_file = filename
    else: 
        dir_path = pathlib.Path(__file__).parent.absolute()
        cfg_file = os.path.join(dir_path, filename)
        assert os.path.isfile(cfg_file), \
               f"{filename} not in configs ({dir_path})"

    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # Basic validations..
    

    print(f" done.")
    return cfg
    