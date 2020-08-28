from grid_tensorflow.graph import Graph
import unittest
from grid_tensorflow.loads import Loads
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from grid_tensorflow.main import Prediction

tf.compat.v1.disable_eager_execution()


class GridTests(unittest.TestCase):

    def testUni(self):
        g = Graph(
            servicepoint_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/servicepoint_edges.csv',
            transformers_edges_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/transformers_edges.csv',
            conf_path='/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json'
        )
        distribution = [0.05] * 5
        duration = 5
        n = 10000
        conf = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/switches.json"
        series = "/home/patrizio-guagliardo/PycharmProjects/energy/src/test/grid/data/timeseries.json"

        import time
        start_time = time.time()
        path = '/home/patrizio-guagliardo/Desktop/Fuses/fuses.ckpt'
        output=Prediction(graph=g, conf=conf, distribution=distribution, duration=duration, n=n, years=[5], series=series)\
            .run(fuse='L1233', path=path)
        print(output)
        print("--- 6: %s seconds ---" % (time.time() - start_time))

        cl = Loads(path=series, conf=conf)
        series_matrix = cl.get_all(col="PROFILEID", loads="LOADVALUE_list", timesteps=4)

        plt.plot(numpy.arange(0, 8760), output)
        #plt.show()
        #plt.savefig('series')

        ye=g.graph.simulation(
            fuse='L1233',
            conf=conf,
            distribution=distribution,
            duration=duration,
            n=n,
            years=[5]
        ).current_time_series(series_matrix)

        current=tf.compat.v1.Session().run(ye)

        plt.plot(numpy.arange(0, 8760), current.reshape((8760, )))
        plt.show()
        plt.savefig('series')


if __name__ == '__main__':
    unittest.main()


