import json
import tensorflow as tf
import numpy


class Configuration(object):
    def __init__(self, switch):
        self._switch = switch

    @property
    def switch(self):
        with open(self._switch) as json_file:
            return json.load(json_file)

    @property
    def one_hot(self):
        keys = self.switch.keys()
        zeros = numpy.identity(len(keys))
        return dict(zip(keys, zeros))