import json
from grid.Configuration import Configuration
import numpy as np


class Loads(object):
    """
    This class aims to provide the time series information of the different profile loads in a matrix
    """
    def __init__(self, path: str, conf):
        self.path = path
        self.conf_ = conf

    @property
    def conf(self):
        return Configuration(self.conf_)

    @property
    def __data__(self):
        data = []
        with open(self.path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_unique(self, name: str, col: str, loads: str, timesteps: int = 1):
        load = np.array([convertion(x) for x in [x[loads] for x in self.__data__ if x[col]==name][0]])
        return load[::timesteps].copy()

    def get_all(self, col: str, loads: str, timesteps: int = 1):
        return np.array([self.get_unique(key, col, loads, timesteps) for key in self.conf.switch.keys()])


def convertion(x):
    try:
        return float(x)
    except:
        raise Exception("This did not work: "+x+"")