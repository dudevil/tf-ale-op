# Run op for 100 times and save all the frames as images

import tensorflow as tf
from PIL import Image

ale_module = tf.load_op_library('build/libaleop.so')

session = tf.Session()

a = tf.placeholder(tf.int32)
step = ale_module.ale(a, "pong.bin", frameskip_min=2, frameskip_max=5)

for i in range(100):
    reward, done, screen = session.run(step, feed_dict={a: 3})
    im = Image.fromarray(screen)
    im.save("frame%d.png"%(i,))

    if done:
        break
    
