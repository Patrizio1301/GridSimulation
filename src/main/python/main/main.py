import tensorflow as tf
from grid.loads import Loads
from analytics.storage import Storage


class Prediction(object):

    def __init__(self, graph, conf, distribution, duration, n, years, series):
        self.graph = graph
        self.conf = conf
        self.distribution = distribution
        self.duration = duration
        self.n = n
        self.years = years
        self.series = series

    def load_simulation(self, sess, fuse, path):
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
        return simulation.time_series_analysis(sess, series, num=8760, path=path, fuse=fuse)

    @staticmethod
    def to_json(sess, variable, fuse: str, path: str, path_json: str):
        fuse_data = Storage.load_fuse(fuse_var=fuse, sess=sess, path=path, need=False, variable=variable)
        Storage.to_json(sess, fuse_data, path_json)

    def run(self, path: str, json_directory: str):
        fuses = self.graph.graph.get_fuses()
        for fuse in fuses:
            with tf.compat.v1.Session() as sess:
                variable = self.load_simulation(sess, fuse, path)
                self.to_json(sess, variable, fuse, path, json_directory+fuse + '.json')