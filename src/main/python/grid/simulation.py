from typing import List
import numpy
from grid.Servicepoint import Servicepoint
from grid.Configuration import Configuration
import tensorflow as tf
from utils.utils import Utils, Storage


class Simulation(object):

    def __init__(self, configuration: str, consumptions: List[Servicepoint], distribution: List[float],
                 duration: int, n: int, years= List[int]):
        self.conf_ = configuration
        self.consumptions = consumptions
        self.distribution = distribution
        self.duration = duration
        self.years = years
        self.n = n

    @property
    def conf(self):
        """Simulation configuration

        :return: <class: Configuration>
        """
        return Configuration(self.conf_)

    @property
    def consumption_tensors(self):
        return tf.stack([consumption.to_tensor for consumption in self.consumptions])

    def current_time_series(self, timeserie):
        """Output is current time serie data.

        :param timeserie:
        :return:
        """

        @tf.function
        def service_point_current(input):
            return input[0]

        distribution=tf.map_fn(
            fn=service_point_current,
            elems=self.consumption_tensors,
            dtype=tf.int64,
            parallel_iterations=100
        )
        current_distribution = tf.expand_dims(tf.cast(tf.reduce_sum(distribution, axis=0), tf.float64), 0)
        return tf.matmul(current_distribution, tf.convert_to_tensor(timeserie))

    @property
    @tf.function
    def simulation(self):
        """Multiple simulations for one single fuse which is defined solely by
        the aggregation service-point, represented in this class by the parameter
        'consumption'. The output will be a list of multiple simulations aggregated
        on load profile type.

        :return: <class. tf.Tensor>
        """

        return tf.map_fn(
            fn=lambda x: self.fuse_simulation,
            elems=tf.range(self.n),
            dtype=tf.int64,
            parallel_iterations=100)

    @property
    @tf.function
    def fuse_simulation(self):
        """One simulation for a determined fuse.

        :return: <class: tf.Tensor>

        >>> years = List[2]
        >>> switch = tf.constant(1)
        >>> current_load_sp1 = tf.convert_to_tensor(numpy.array([1, 0, 0, 0, 0, 0]))
        >>> future_load_sp1 = tf.convert_to_tensor(numpy.array([0, 1, 0, 0, 0, 0]))
        >>> current_load_sp2 = tf.convert_to_tensor(numpy.array([0, 0, 1, 0, 0, 0]))
        >>> future_load_sp2 = tf.convert_to_tensor(numpy.array([0, 0, 0, 1, 0, 0]))
        >>> consumption_tensors = [tf.tuple(current_load_sp1, future_load_sp1), tf.tuple(current_load_sp2, future_load_sp2)]
        >>> fuse_simulation()
        <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
          array([[0, 1, 0, 1, 0, 0]], dtype=int32)>

        """
        dist = self.distribution + [1 - sum(self.distribution)]
        samples = tf.compat.v1.multinomial(tf.compat.v1.log([dist]), len(self.consumptions))  # note log-prob
        distribution_indices = tf.cast(tf.squeeze(samples), tf.int64)

        return tf.map_fn(
            fn=self.service_point_simulation,
            elems=(distribution_indices, self.consumption_tensors),
            fn_output_signature=tf.int64,
            parallel_iterations=10)

    @tf.function
    def service_point_simulation(self, input):
        """

        :param input: Tuple (switch, consumption_tensor) where consumption_tensor is a <class: tf.tuple>
        with two elements.
        :return:

        ## Example:
        >>> years = List[2]
        >>> switch = tf.constant(1)
        >>> current_load = tf.convert_to_tensor(numpy.array([1, 0, 0, 0, 0, 0]))
        >>> future_load = tf.convert_to_tensor(numpy.array([0, 1, 0, 0, 0, 0]))
        >>> consumption_tensor = tf.tuple(current_load, future_load)
        >>> service_point_simulation((switch, consumption_tensor))
        <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
          array([[0, 1, 0, 0, 0, 0]], dtype=int32)>

        If the switch year is bigger than 2, in year 2 the current load is still used. Thus:
        >>> years = List[2]
        >>> switch = tf.constant(3)
        >>> current_load = tf.convert_to_tensor(numpy.array([1, 0, 0, 0, 0, 0]))
        >>> future_load = tf.convert_to_tensor(numpy.array([0, 1, 0, 0, 0, 0]))
        >>> consumption_tensor = tf.tuple(current_load, future_load)
        >>> service_point_simulation((switch, consumption_tensor))
        <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
          array([[1, 0, 0, 0, 0, 0]], dtype=int32)>

        """
        current_count = [tf.cast(tf.less(tf.constant([i], tf.int64), input[0] + 1), tf.int32) for i in self.years]
        current_count = tf.reduce_sum(current_count, 0)
        future_count = tf.constant(len(self.years)) - current_count
        return tf.concat(
            [tf.tile(tf.expand_dims(input[1][0], 0), tf.concat([current_count, tf.constant([1])], 0)),
             tf.tile(tf.expand_dims(input[1][1], 0), tf.concat([future_count, tf.constant([1])], 0))], 0)

    def get_time_series(self, timeserie, simulations):
        """

        :param timeserie:
        :param simulations:
        :param years:
        :return:
        """
        simulations_years = tf.gather(simulations, tf.convert_to_tensor(numpy.array(self.years)), axis=1)
        simulations_years_new = tf.cast(simulations_years, tf.float64)
        return tf.reshape(tf.matmul(simulations_years_new, tf.convert_to_tensor(timeserie)), (self.n, -1))

    @staticmethod
    def time_series_analysis(sess, series: tf.Tensor, num: int, path: str, fuse: str):
        """

        :param series:
        :param num:
        :param path:
        :param fuse:
        :return:
        """
        unstacked = tf.unstack(tf.transpose(series), num=num)
        num = 2190
        dataset = tf.data.Dataset.from_tensor_slices(unstacked).map(
            lambda x: tf.stack([tf.math.reduce_min(x, axis=0), tf.math.reduce_mean(x, axis=0), tf.math.reduce_max(x, axis=0)])
        ).batch(num)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        next_element = iterator.get_next()
        series_var = tf.Variable(numpy.zeros([8760, 3]), name=fuse)
        values = tf.reshape(next_element, (num * 3,))
        saver = tf.compat.v1.train.Saver()

        sess.run(iterator.initializer)
        sess.run(tf.compat.v1.global_variables_initializer())
        print(range(int(8760 / num)))
        for i in range(int(8760 / num)):
            indices = Utils.indices(i * num, i * num + num, 0, 3)
            sparse = tf.SparseTensor(indices=tf.cast(indices, tf.int64), values=values, dense_shape=[8760, 3])
            sparse_ = tf.sparse.add(series_var, sparse)
            _ = sess.run(tf.compat.v1.assign(series_var, sparse_))
        Storage.save(sess, saver, path, 1)
        return series_var

