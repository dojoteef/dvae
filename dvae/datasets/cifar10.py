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

"""A simple CIFAR10 data loader based on code from Tensorflow. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip # pylint: disable=unused-import
import os
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.datasets.dataloader import Data
from dvae.datasets.dataloader import Dataset
from dvae.datasets.dataloader import DataLoader


class CIFAR10DataLoader(DataLoader):
    """ Class for loading CIFAR data. """
    PIXEL_DEPTH = 255
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGES_PER_FILE = 10000

    SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/'
    SOURCE_FILENAME = 'cifar-10-binary.tar.gz'
    SOURCE_DIRECTORY = 'cifar-10-batches-bin/'

    def __init__(self, datadir):
        super(CIFAR10DataLoader, self).__init__(datadir)

    def maybe_download(self):
        """Download the data from Alex Krizhevsky's website, unless it's already here."""
        if not tf.gfile.Exists(self.datadir):
            tf.gfile.MakeDirs(self.datadir)
        filepath = os.path.join(self.datadir, CIFAR10DataLoader.SOURCE_FILENAME)
        if not tf.gfile.Exists(filepath):
            url = CIFAR10DataLoader.SOURCE_URL + CIFAR10DataLoader.SOURCE_FILENAME
            filepath, _ = urllib.request.urlretrieve(url, filepath)
            with tf.gfile.GFile(filepath) as datafile:
                size = datafile.size()
            print('Successfully downloaded', size, 'bytes.')
        return tarfile.open(filepath, 'r:gz')

    def extract_images_and_labels(self, archive, filenames):
        """Extract the images into a 4D tensor [image index, y, x, channels].

        Values are rescaled from [0, 255] down to [0.0, 1.0].

        Extract the labels into a vector of int64 label IDs.
        """
        filenames = list(filenames)
        count = CIFAR10DataLoader.IMAGES_PER_FILE * len(filenames)

        # Source image format is [image index, channels, y, x]
        image_dimensions = (
            3,
            CIFAR10DataLoader.IMAGE_WIDTH,
            CIFAR10DataLoader.IMAGE_HEIGHT)

        images = np.zeros((count,) + image_dimensions)
        labels = np.zeros(count)

        for index, filename in enumerate(filenames):
            print('Extracting', filename)
            filepath = os.path.join(CIFAR10DataLoader.SOURCE_DIRECTORY, filename)
            with archive.extractfile(filepath) as bytestream:

                # Read labels and images
                record_count = CIFAR10DataLoader.IMAGES_PER_FILE * (np.prod(image_dimensions) + 1)
                buf = bytestream.read(np.sum(record_count))

                dtype = [('labels', np.uint8), ('images', np.uint8, image_dimensions)]
                data = np.frombuffer(buf, dtype=dtype)
                data = data.view(np.recarray)

                start = index * CIFAR10DataLoader.IMAGES_PER_FILE
                end = start + CIFAR10DataLoader.IMAGES_PER_FILE
                labels[start:end] = data.labels
                images[start:end] = data.images

        # Convert from source image format to [image index, y, x, channels] format
        images = np.transpose(images, [0, 2, 3, 1])
        images = images.astype(np.float32)
        images = images / CIFAR10DataLoader.PIXEL_DEPTH
        return images, labels

    def load_data(self, validation_percent=0.1, **kwargs):
        """ Load the data used for training/validation/testing."""
        # Get the data.
        archive = self.maybe_download()
        train_files = ('data_batch_{0}.bin'.format(i+1) for i in xrange(0, 5))
        test_files = ('test_batch.bin',)

        # Extract it into np arrays.
        test = Data(Data.TEST, *self.extract_images_and_labels(archive, test_files))
        train = Data(Data.TRAIN, *self.extract_images_and_labels(archive, train_files))

        # Generate a validation set.
        count = int(validation_percent * len(train.images))
        validation = Data(Data.VALIDATE, train.images[:count, ...], train.labels[:count])
        train.images = train.images[count:, ...]
        train.labels = train.labels[count:]

        return Dataset(test, train, validation)
