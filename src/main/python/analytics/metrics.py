import tensorflow as tf
import tensorflow_probability as tfp


class Metrics(object):

    @staticmethod
    def positives(serie: tf.Tensor):
        return tf.compat.v1.count_nonzero(tf.greater_equal(serie, 0.))

    @staticmethod
    def negatives(serie: tf.Tensor):
        return tf.compat.v1.count_nonzero(tf.less_equal(serie, 0.))

    @staticmethod
    def negatives_percentage(serie: tf.Tensor):
        s=tf.transpose(serie)
        negatives = tf.compat.v1.count_nonzero(tf.less_equal(s, 0.), 1)
        positives = tf.compat.v1.count_nonzero(tf.greater_equal(s, 0.), 1)
        return negatives/(negatives+positives)

    @staticmethod
    def positive_percentage(serie: tf.Tensor):
        negatives = tf.compat.v1.count_nonzero(tf.less_equal(serie, 0.))
        positives = tf.compat.v1.count_nonzero(tf.greater_equal(serie, 0.))
        return positives/(negatives+positives)

    @staticmethod
    def negative_values(serie: tf.Tensor):
        asserts = tf.less_equal(serie, tf.constant(0, tf.float64))
        return tf.gather_nd(serie, tf.where(asserts))

    @staticmethod
    def negative_variance(serie: tf.Tensor):
        asserts = tf.less_equal(serie, tf.constant(0, tf.float64))
        return tfp.stats.variance(tf.gather_nd(serie, tf.where(asserts)))

    @staticmethod
    def max(serie: tf.Tensor):
        return tf.reduce_max(serie, axis=0)

    @staticmethod
    def min(serie: tf.Tensor):
        return tf.reduce_min(serie, axis=0)