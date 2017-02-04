# Copyright 2016 The Nader Akoury. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Image utilities """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def tile_images(images, tiles=64):
    """
    Combine the passed in images into a single image with tiles * tiles images.

    Arguments:
        images - a tensor of images to tile of shape [batch, rows, columns, depth]
        tiles - how many total tiles

    Result:
        A zero padded tensor [1, tiles * rows, tiles * columns, depth] of image tiles
    """
    # Make a rectangular shape that holds the tiled images.
    shape = tf.shape(images)
    width = int(np.floor(np.sqrt(tiles)))
    height = int(np.ceil(tiles/width))

    image_size = width * height

    # How many blank images to pad
    tile_count = tf.reduce_min([shape[0], tiles])
    blanks = tf.reduce_max([image_size - tile_count, 0])
    padded = tf.pad(images[:tile_count], tf.stack([[0, blanks], [0, 0], [0, 0], [0, 0]]))

    # Finally reshape the images to be tiled
    rows = tf.concat(1, (tf.split(0, image_size, padded)))
    columns = tf.concat(2, (tf.split(1, width, rows)))

    return columns
