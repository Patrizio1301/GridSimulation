from utils.utils import Storage as sa
import tensorflow as tf
import json
from analytics.metrics import Metrics


class Storage(object):

    @staticmethod
    def load_fuse(fuse_var, sess, path, need=False, variable=None):
        if need:
            tensor = tf.compat.v1.get_variable(fuse_var, shape=[3], dtype=tf.float64)
        saver = tf.compat.v1.train.Saver()
        sa.loadmodel(sess, saver, path)
        if need:
            return tensor
        else:
            return variable

    @staticmethod
    def to_json(sess: tf.compat.v1.Session, tensor: tf.Tensor, path: str):
        print("HEEEEEEEEEEEEEEEEE", sess.run(tensor))
        max = sess.run(Metrics.max(tensor))
        print("MAAAAAAAAAAAAAX", max)
        min = sess.run(Metrics.min(tensor))
        print("IIIIIIIIIIIIIIIIIIIIIIIN", min)
        neg_perc = sess.run(Metrics.negatives_percentage(tensor))
        print("NEEEEEEEEEEE", neg_perc)
        data = {}
        data['min_serie'] = {
            'max': max[0],
            'min': min[0],
            'negative_percentage': neg_perc[0]
        }
        data['mean_serie'] = {
            'max': max[1],
            'min': min[1],
            'negative_percentage': neg_perc[1]
        }
        data['max_serie'] = {
            'max': max[2],
            'min': min[2],
            'negative_percentage': neg_perc[2]
        }

        with open(path, 'w') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent = 4)
            outfile.write('\n')

