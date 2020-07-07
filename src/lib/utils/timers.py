""" timers.py (By: Charley Zhang, July 2020)
Tools to keep track of running-time and print them neatly.
"""


import time
from collections import OrderedDict

class StopWatch:

    def __init__(self):
        self.reset()

    def tic(self, name='default'):
        self.stops[name] = 0
        self.starts[name] = time.time()
        return self

    def toc(self, name='default', disp=True):
        if name not in self.starts:
            print(f"StopWatch({name}) did not get tic'd yet.")
            return
        
        self.stops[name] = time.time()
        if disp:
            self.disp(name=name)
        return self.stops[name] - self.starts[name]

    def disp(self, name='default'):
        if name not in self.stops or name not in self.starts:
            print(f"StopWatch({name}) did not get tic'd yet.")
            return
        
        elapsed = self.stops[name] - self.starts[name]
        if elapsed > 0:
            if elapsed > 60:
                print(f"StopWatch({name}) took {elapsed/60:.2f} min")
            else:
                print(f"StopWatch({name}) took {elapsed:.2f} sec")
        else:
            print(f"StopWatch({name}) did not get toc'd for its tic.")

    def reset(self):
        self.starts = {}  # time in seconds
        self.stops = {}
        return self

        