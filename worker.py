import tensorflow as tf
import threading
import numpy as np
import time

ale_module = tf.load_op_library('build/libaleop.so')

class AleThread(threading.Thread):
    pass

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
for _ in range(20):
    t = threading.Thread(target=env_runner, args=(sess, coord, enqueue_op, action))
    threads.append(t)
    
 
for t in threads:
    t.start()

time.sleep(2)
c = 0
stime = time.time()
while c < 500000:
    c += sess.run(dequeue_op)[0].shape[0]
    
print("Got %d in %f seconds" % (c, time.time() - stime))
    
#     print("Got: " +str(r[0].shape))
# for _ in range(3):
#     time.sleep(1)

#     print(sess.run(queue.size()))
#     r = sess.run(dequeue_op)
#     print("Got: " +str(r[0].shape))

coord.request_stop()
coord.join(threads)
