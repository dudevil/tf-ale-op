# TF-Ale-Op
TF-Ale-Op is a TensorFlow Op that makes it possible to include the [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) into the computational graph without the overhead of dealing with python interface and `feed_dict`. 

It accepts a single scalar tensor as action and returns a reward, end-of-episode flag, observation tuple. The Op is thread-safe and multiple environments can be run in parallel.


## Build instructions
- install [TensorFlow](https://www.tensorflow.org/install/)
- clone the [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) repository
- compile and install ALE:

```bash
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
cd Arcade-Learning-Environment
mkdir build && cd build
```
If you installed tensorflow binary package *and* are using GCC version 5 or later run:
```bash
cmake -D CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
```
If you compiled tensorflow from source or are using GCC version 4 run:
```bash
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
```

```bash
make -j && make install
```

- clone this repository
- `python setup.py install`

## Basic usage example

```python

import tensorflow as tf
from aleop import ale

session = tf.Session()

a = tf.placeholder(tf.int32)
step = ale(a, "pong", frameskip_min=2, frameskip_max=5)

reward, done, screen = session.run(step, feed_dict={a: 3})
```
