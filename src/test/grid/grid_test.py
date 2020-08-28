from grid_tensorflow.graph import Graph
import unittest
from grid_tensorflow.loads import Loads
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from grid_tensorflow.grid import Grid
import networkx as nx

tf.compat.v1.disable_eager_execution()


class GridTests(unittest.TestCase):
    path = './data/graph_test.graphml'
    graph = Grid(nx.read_graphml(path, str, int))

    def test_get_terminals(self):
        terminals = self.graph.get_transformers_from_fuse('1')
        servicepoints = self.graph.get_servicepoint_from_transformer('3')
        servicepoint_fuse = self.graph.get_servicepoints_from_fuse('1')
        self.assertEqual([x.id for x in terminals], ['2', '3'])
        self.assertEqual([x.id for x in servicepoints], ['6', '7'])
        self.assertEqual([x.id for x in servicepoint_fuse], ['4', '5', '6', '7'])

    def test_simulation(self):
        distribution = [0.05] * 5
        duration = 5
        n = 10000
        conf = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json"
        series = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/timeseries.json"
        sim = self.graph.simulation(
            fuse='',
            conf=conf,
            distribution=distribution,
            duration=duration,
            n=n,
            years=['5']
        )
        print("sim")


if __name__ == '__main__':
    unittest.main()


