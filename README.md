# tf-ale-op

## Build instructions
- Install Tensorflow
- Download and compile the [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- Download and save the Atari rom files
- Clone this repository
- `mkdir build && cd build`
- `cmake .. -DALE_INCLUDE_DIR=ALE_PATH/src/ -DALE_LIBRARY_DIR=ALE_PATH -DROM_PATH=PATH_TO_ROM_FILES`
- `make`

The build should produce the `libaleop.so` file.

## Usage example

```python

import tensorflow as tf

ale_module = tf.load_op_library('build/libaleop.so')

session = tf.Session()

a = tf.placeholder(tf.int32)
step = ale_module.ale(a, "pong.bin", frameskip_min=2, frameskip_max=5)

reward, done, screen = session.run(step, feed_dict={a: 3})
```
