import unittest
import tensorflow as tf
from analytics.storage import Storage
from analytics.metrics import Metrics
tf.compat.v1.disable_eager_execution()


class StorageTests(unittest.TestCase):

    def testLoad(self):
        path = '/home/patrizio-guagliardo/Desktop/Fuses/fuses.ckpt'
        with tf.compat.v1.Session() as sess:
            tens=Storage.load_fuse(fuse_var='time_series', sess=sess, path=path)
            # Check the values of the variables
            print(Metrics.max(tens).eval())
            Storage.to_json(sess, tens, 'json.json')


if __name__ == '__main__':
    unittest.main()
