import unittest
import tensorflow as tf
from utils.utils import Storage

tf.compat.v1.disable_eager_execution()


class UtilsTests(unittest.TestCase):

    def testLoad(self):
        tensor = tf.compat.v1.get_variable('time_series', shape=[3], dtype=tf.float64)
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            Storage.loadmodel(session=sess, saver=saver, checkpoint_dir='../grid/model.ckpt')
            # Check the values of the variables
            print("v1 : %s" % tensor.eval())


if __name__ == '__main__':
    unittest.main()
