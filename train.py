import tensorflow as tf
from tensorflow.python.framework import constant_op
import collections
import threading
import numpy as np
import time
from model import _process_frame, FFPolicy

ale_module = tf.load_op_library('build/libaleop.so')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'):
    policy = FFPolicy(state, 4)

with tf.device('/cpu:0'):
    queue = tf.FIFOQueue(
        capacity=1024, 
        dtypes=[tf.float32, tf.bool, tf.float32], 
        shapes=[(), (), (42, 42, 1)])

    action = tf.Variable(0, dtype=tf.int32, trainable=False, name="action") # tf.random_uniform((), maxval=4, dtype=tf.int32)
    reward, done, state = ale_module.ale(action, "pong.bin", frameskip_min=4, frameskip_max=4)
    n_action, value = policy.act(tf.expand_dims(state, [0]))
    action.assign(n_action)
    enqueue_op = queue.enqueue((reward, done, _process_frame(state)))
    dequeue_op = queue.dequeue_up_to(64)

def env_runner(sess, coord, enqueue_op, action):
    while not coord.should_stop():
        try:
            sess.run(enqueue_op)
        except tf.errors.CancelledError:
            print("Cancelled")
            return
    print("Worker stopping")


coord = tf.train.Coordinator()
threads = []
for _ in range(10):
    t = threading.Thread(target=env_runner, args=(sess, coord, enqueue_op, action))
    threads.append(t)
    
 
for t in threads:
    t.start()

reward, action, state = dequeue_op

sess.run(tf.initialize_all_variables())
    
c = 0
stime = time.time()
while c < 100000:
    #c += [0].shape[0]
    c += sess.run(policy.sample).shape[0]  
    
print("Got %d in %f seconds" % (c, time.time() - stime))
    
coord.request_stop()
coord.join(threads)
