import tensorflow as tf
from grid.loads import Loads


class Prediction(object):

    def __init__(self, graph, conf, distribution, duration, n, years, series):
        self.graph = graph
        self.conf = conf
        self.distribution = distribution
        self.duration = duration
        self.n = n
        self.years = years
        self.series = series

    def run(self, fuse, path):
        """ execute and storage one fuse simulation """

        simulation = self.graph.graph.simulation(
            fuse=fuse,
            conf=self.conf,
            distribution=self.distribution,
            duration=self.duration,
            n=self.n,
            years=self.years)

        cl = Loads(path=self.series, conf=self.conf)
        series_matrix = cl.get_all(col="PROFILEID", loads="LOADVALUE_list", timesteps=4)
        sim = tf.math.reduce_sum(simulation.simulation, axis=1)
        simulations_years_new = tf.cast(sim, tf.float64)
        series = tf.reshape(tf.matmul(simulations_years_new, tf.convert_to_tensor(series_matrix)), (self.n, -1))
        return simulation.time_series_analysis(series, num=8760, path=path)