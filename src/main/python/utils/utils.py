import tensorflow as tf
import numpy as np
import os


class Utils(object):

    @staticmethod
    def indices(column_start, column_end, row_start, row_end):
        row_length=row_end-row_start
        columns = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(column_start, column_end), 1), 2), tf.constant([1, row_length, 1]))
        index = tf.expand_dims(tf.expand_dims(tf.constant(np.array(range(row_start, row_end)), tf.int32), 1), 0)
        rows = tf.tile(index, tf.constant([column_end-column_start, 1, 1]))
        return tf.reshape(tf.concat([columns, rows], axis=2), (-1, 2))


class Storage(object):

    @staticmethod
    def loadmodel(session, saver, checkpoint_dir):
        session.run(tf.compat.v1.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    @staticmethod
    def save(session, saver, checkpoint_dir, step):
        dir = os.path.join(checkpoint_dir, "model")
        saver.save(session, dir, global_step=step)


