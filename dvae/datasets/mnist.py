# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A simple MNIST data loader based on code from Tensorflow. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np
from six.moves import urllib
import tensorflow as tf

from dvae.datasets.dataloader import Data
from dvae.datasets.dataloader import Dataset
from dvae.datasets.dataloader import DataLoader


class MNISTDataLoader(DataLoader):
    """ Class for loading MNIST data. """
    PIXEL_DEPTH = 255
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, datadir):
        super(MNISTDataLoader, self).__init__(datadir)

    def read_header_int(self, bytestream):
        """ Read an int32 from the header. For file format see SOURCE_URL """
        msb = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=msb)[0]

    def maybe_download(self, filename):
        """Download the data from Yann's website, unless it's already here."""
        if not tf.gfile.Exists(self.datadir):
            tf.gfile.MakeDirs(self.datadir)
        filepath = os.path.join(self.datadir, filename)
        if not tf.gfile.Exists(filepath):
            url = MNISTDataLoader.SOURCE_URL + filename
            filepath, _ = urllib.request.urlretrieve(url, filepath)
            with tf.gfile.GFile(filepath) as datafile:
                size = datafile.size()
            print('Successfully downloaded', filename, size, 'bytes.')
        return filepath

    def extract_labels(self, filename):
        """Extract the labels into a vector of int64 label IDs."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self.read_header_int(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic for MNIST labels')

            num_labels = self.read_header_int(bytestream)
            buf = bytestream.read(1 * num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def extract_images(self, filename):
        """Extract the images into a 4D tensor [image index, y, x, channels].

        Values are rescaled from [0, 255] down to [0.0, 1.0].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self.read_header_int(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic for MNIST images')

            num_images = self.read_header_int(bytestream)
            rows = self.read_header_int(bytestream)
            columns = self.read_header_int(bytestream)
            image_dimensions = (num_images, rows, columns, 1)

            buf = bytestream.read(np.prod(image_dimensions))
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = 1.0 - data / MNISTDataLoader.PIXEL_DEPTH
            data = data.reshape(*image_dimensions)
            return data

    def load_data(self, validation_percent=0.1, **kwargs):
        """ Load the data used for training/validation/testing."""
        # Get the data.
        train_images_filename = self.maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = self.maybe_download('train-labels-idx1-ubyte.gz')
        test_images_filename = self.maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = self.maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into np arrays.
        train = Data(Data.TRAIN, self.extract_images(train_images_filename),
                     self.extract_labels(train_labels_filename))
        test = Data(Data.TEST, self.extract_images(test_images_filename),
                    self.extract_labels(test_labels_filename))

        # Generate a validation set.
        count = int(validation_percent * len(train.images))
        validation = Data(Data.VALIDATE, train.images[:count, ...], train.labels[:count])
        train.images = train.images[count:, ...]
        train.labels = train.labels[count:]

        return Dataset(test, train, validation)
