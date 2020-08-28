from grid_tensorflow.graph import Graph
import unittest
import tensorflow as tf
from main.main import Prediction

tf.compat.v1.disable_eager_execution()


class GridTests(unittest.TestCase):

    def test_main(self):
        g = Graph(
            servicepoint_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/servicepoint_edges.csv',
            transformers_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/transformers_edges.csv',
            conf_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json'
        )
        distribution = [0.05] * 5
        duration = 5
        n = 50000
        conf = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json"
        series = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/timeseries.json"

        import time
        start_time = time.time()
        path = '/home/patrizio-guagliardo/Desktop/Fuses/fuses.ckpt'
        Prediction(graph=g, conf=conf, distribution=distribution, duration=duration, n=n, years=[5], series=series)\
            .run(path=path, json_directory='/home/patrizio-guagliardo/Desktop/Fuses/results/')
        print("--- 6: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    unittest.main()