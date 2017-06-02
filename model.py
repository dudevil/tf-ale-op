import numpy as np
import tensorflow as tf

def _process_frame(frame):
    frame = tf.image.crop_to_bounding_box(frame, 34, 0, 160, 160)
    frame = tf.expand_dims(frame, [0])
    frame = tf.image.resize_bilinear(frame, (80, 80))
    frame = tf.image.resize_bilinear(frame, (42, 42))
    frame = tf.reduce_mean(frame, axis=3, keep_dims=True)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return frame


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class FFPolicy(object):

    def __init__(self, observation, action_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action_space = action_space
        self.x = x = observation
        for i in range(4):
            x = tf.contrib.layers.conv2d(x,
                                         num_outputs=32,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         activation_fn=tf.nn.elu,
                                         scope="conv%d" % i)

        x = tf.contrib.layers.flatten(x)
        
        self.logits = tf.contrib.layers.fully_connected(x,
                                                        action_space,
                                                        activation_fn=None,
                                                        weights_initializer=normalized_columns_initializer(0.01),
                                                        scope="action")

        self.vf = tf.contrib.layers.fully_connected(x,
                                                    1,
                                                    activation_fn=None,
                                                    weights_initializer=normalized_columns_initializer(0.01),
                                                    # biases_initializer=tf.constant_initializer(-1.),
                                                    scope="value")

        self.sample = categorical_sample(self.logits, action_space)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self):
        return self.sample, self.vf[0]
    
    def value(self):
        return self.vf[0]
