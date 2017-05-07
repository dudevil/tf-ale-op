from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_ale_module = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'aleop.so'))

def _game_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "roms")

def get_game_path(game_name):
    return os.path.join(_game_dir(), game_name) + ".bin"

def ale(action, game_name, **kwargs):
    return _ale_module.ale(action, get_game_path(game_name), **kwargs)
