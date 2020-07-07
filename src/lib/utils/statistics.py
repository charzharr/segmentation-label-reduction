
import numbers
import matplotlib.pyplot as plt
import numpy as np
import collections
import wandb as wab


class ExperimentTracker:
    """ All-in-one concise experiment tracking after each epoch completion.
    - Tracks all essential iteration and epoch metrics.
    - Automatically logs epoch stats to wandb.
    """

    def __init__(self, metric_names=[], wandb=None):
        self.metrics_d = { k: [] for k in metric_names } if metric_names else {}
        if wandb:
            self.use_wandb = True
            self._setup_wandb(wandb)
        else:
            self.use_wandb = False
    
    def update(self, metrics_d, wandb=False, verbose=False):
        for k, item in metrics_d.items():
            if k not in self.metrics_d:
                if verbose:
                    print(f" (StatTracker) Adding stat: {k}, Val: {item}")
                self.metrics_d[k] = [item]
                continue
            self.metrics_d[k].append(item)
            if verbose:
                print(f"(StatTracker) Adding {k}: {item}")
        if wandb and self.use_wandb:
            wab.log(metrics_d)
    
    def best(self, metric_name, max_better=True):
        if metric_name not in self.metrics_d:
            print(f" (ExperimentTracker) Given k({metric_name}) not valid.")
            return None
        metric_hist = self.metrics_d[metric_name]
        return max(metric_hist) if max_better else min(metric_hist)

    def plot(self, keys=None):
        names, vals = [], []
        if keys:
            for k in keys:
                if k not in self.metrics_d:
                    continue
                names.append(k)
                vals.append(self.metrics_d[k])
        else:
            for k, v in self.metrics_d.items():
                if v and not isinstance(v[0], collections.Sequence):
                    names.append(k)
                    vals.append(vals)

        num_cols = 4
        num_rows = len(names) // num_cols + 1
        fig = plt.figure(figsize=(20, 5*num_rows))
        for i, name in enumerate(names):
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_title(name)
            ax.plot(vals[i])
        plt.show()
            
    def _setup_wandb(self, cfg):
        print(" > Initializing Weights and Biases run..")
        name = cfg['experiment']['id'] + '_' + cfg['experiment']['name']
        wab.init(
            name=name, 
            project=cfg['experiment']['project'],
            config=cfg,
            notes=cfg['experiment']['description']
        )


class EpochMeters:
    """ Updates every iteration and keeps track of accumulated stats """
    
    def __init__(self):
        self.accums = {}
        self.ns = {}

    def update(self, metrics_d, n=1):
        for k, item in metrics_d.items():
            if k not in self.accums:
                self.accums[k] = item
                self.ns[k] = n
                continue
            self.accums[k] += item
            self.ns[k] += n

    def avg(self, no_avg=[]):
        ret = {}
        for k, v in self.accums.items():
            if k in no_avg:
                ret[k] = v
            else:
                ret[k] = v/self.ns[k]
        return ret


class BaseStatTracker:

    def __init__(self, stat_names=None):
        self.stats_dict = {k: [] for k in stat_names} if stat_names else {}
    
    def update(self, update_dict, disp=False):
        for k, item in update_dict.items():
            if k not in self.stats_dict:
                print(f"(StatTracker) Adding stat: {k}, Val: {item}")
                self.stats_dict[k] = [item]
                continue
            self.stats_dict[k].append(item)
            if disp:
                print(f"(StatTracker) Adding {k}: {item}")

    def reset(self, keys=None):
        for sk in self.stats_dict:
            if not keys:
                print(f"(StatTracker) Wiping stat: {sk}")
                self.stats_dict[sk] = []
            else:
                for k in keys:
                    if k in sk:
                        print(f"(StatTracker) Wiping stat: {sk}")
                        self.stats_dict[sk] = []
    
    def print_info(self, print_item=True):
        print(f"(StatTracker) Printing info for {len(self.stats_dict)} stats..")
        for k, v in self.stats_dict.items():
            print(f"   > {k} ({len(v)} items ", end='') 
            if len(v) > 0:
                print(f"of type {type(v)})")
                print('    ', v)

    def plot(self, keys=None):
        names, vals = [], []
        if keys:
            for k in keys:
                if k not in self.stats_dict:
                    continue
                names.append(k)
                vals.append(self.stats_dict[k])
        else:
            for k, v in self.stats_dict.items():
                if v and not isinstance(v[0], collections.Sequence):
                    names.append(k)
                    vals.append(vals)

        num_cols = 4
        num_rows = len(names) // num_cols + 1
        fig = plt.figure(figsize=(20, 5*num_rows))
        for i, name in enumerate(names):
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_title(name)
            ax.plot(vals[i])
        plt.show()

        
class AverageMeter:
    
    def __init__(self, init_val=None):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n == 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            

class NAverageMeter:
    
    def __init__(self, n=1):
        self.n = n
        self.reset()

    def reset(self):
        self.avg = np.array([0.]*self.n)
        self.sum = np.array([0.]*self.n)
        self.count = 0

    def update(self, val, n=1):
        if n == 0:
            self.sum += np.array(val)
            self.count += 1
            self.avg = self.sum
        else:
            self.sum += np.array(val) * n
            self.count += n
            self.avg = self.sum / self.count



# Basic Tests
if __name__ == '__main__':
    import sys

    for epoch in range(1, 20):
        epmeter = EpochMeters()
        for it in range(1, 5):
            epmeter.update({
                'TPs': np.array([1,2,3]),
                'F1': 1.1,
                'F1s': np.array([.2,.4,.6])
            })
        print(epmeter.avg(no_avg=['TPs']))
        sys.exit(0)
            
