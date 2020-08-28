from grid.Configuration import Configuration
import tensorflow as tf
from typing import List


class Servicepoint(object):
    """
    ARGS:
    --------------
    current_load:
        current load profile type
    future_load:
        future load profile type
    switch_year: int
        the first year, where the future load profile should be provided
    duration: int
        number of future years the simulation should be provided


    METHODS:
    ----------------
    switch_year:

    """
    def __init__(self, current_load: str, conf: Configuration):
        self.current_load = current_load
        self.conf = conf
        self.switch_conf = conf.switch[current_load]

    @property
    def switch(self):
        return self.switch_conf['switch']

    @property
    def future_load(self):
        if self.switch:
            return self.switch_conf['load']
        else:
            return self.current_load

    @staticmethod
    def current_count(switch_year, years):
        return sum(i < switch_year+1 for i in years)

    def load_dist(self, switch_year, years=List[int]):
        current_count=self.current_count(switch_year, years)
        if self.future_load:
            dist = [tf.convert_to_tensor(self.conf.one_hot[self.current_load])]*current_count\
                   +[tf.convert_to_tensor(self.conf.one_hot[self.future_load])]*(len(years)-current_count)
        else:
            dist = [tf.convert_to_tensor(self.conf.one_hot[self.current_load])] * len(years)
        return tf.convert_to_tensor(dist)

    @property
    def to_tensor(self):
        return tf.tuple([tf.convert_to_tensor(self.conf.one_hot[self.current_load], tf.int64),
                         tf.convert_to_tensor(self.conf.one_hot[self.future_load], tf.int64)])