#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re
import os
import sys
import tensorflow as tf

def pkgconfig(*packages):
    return os.popen("pkg-config --libs --cflags %s" % ' '.join(packages)).read().rstrip()

# Compile the bespoke TensorFlow ops in-place. Not sure how this would work if this script wasn't executed as `develop`.
tf_include = tf.sysconfig.get_include()
ale_inc_libs = pkgconfig("ale")
if not ale_inc_libs:
    sys.exit("ALE was not found by pkg-confing. Are sure you insalled ALE?")

compile_command = "g++ -std=c++11 -shared aleop/ale.cc -o aleop/aleop.so " \
                  "-Wno-deprecated-declarations -fPIC -O3 -fomit-frame-pointer " \
                  + ale_inc_libs + " -I{}".format(tf_include)
if sys.platform == "darwin":
    # Additional command for Macs, as instructed by the TensorFlow docs
    compile_command += " -undefined dynamic_lookup"
elif sys.platform.startswith("linux"):
    gcc_version = int(re.search('\d+.', os.popen("gcc --version").read()).group()[0])
    if gcc_version > 4:
        compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0"

print(compile_command)
exit_code = os.system(compile_command) >> 8
if exit_code != 0:
    sys.exit("Failed to compile the ALE Op, please cheak gcc output.")

setup(name='AleOp',
      version='0.1.0',
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/dudevil/tf-ale-op",
      package_data={'aleop': ['aleop.so', 'roms/*']},
      include_package_data=True,
      ext_modules=[],
      packages=["aleop"],
      package_dir={'aleop': 'aleop'},
      py_modules=['aleop.__init__'],
      install_require={'tensorflow': ['tensorflow>=1.0.0'],
                      'tensorflow with gpu': ['tensorflow-gpu>=1.0.0']},
      )
