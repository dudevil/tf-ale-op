import tensorflow as tf
import numpy as np
from worker import  AleThread
import time

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=True


    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    with sess.as_default():
        queue = tf.FIFOQueue(100, dtypes=(tf.float32, tf.float32, tf.float32, tf.float32))
        sw = tf.summary.FileWriter('.')

        at = AleThread(queue, coord, sess, sw)

        sess.run(tf.global_variables_initializer())

        at.start()

        time.sleep(60)

    coord.request_stop()
    coord.join([at])
