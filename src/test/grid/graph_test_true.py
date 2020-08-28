from grid_tensorflow.graph import Graph
import unittest
from grid_tensorflow.loads import Loads
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from grid_tensorflow.main import Prediction
tf.compat.v1.disable_eager_execution()


class GridTests(unittest.TestCase):

    def test_construction(self):
        g = Graph(
            servicepoint_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/servicepoint_edges.csv',
            transformers_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/transformers_edges.csv',
            conf_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json'
        ).graph


        print(g.get_fuses())


if __name__ == '__main__':
    unittest.main()


