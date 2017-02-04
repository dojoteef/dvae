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

import numpy as np


class DataLoader(object):
    """ Class for manipulating the data. """
    def __init__(self, datadir):
        self.datadir = datadir

    def load_data(self, **kwargs):
        """ Load the data used for training/validation/testing."""
        raise NotImplementedError


class Data(object):
    """ A class that groups the data for a particular data grouping (testing, training, etc). """
    TEST = 'testing'
    TRAIN = 'training'
    VALIDATE = 'validation'

    def __init__(self, collection, images, labels):
        self.collection = collection
        self.index = None
        self.images = images
        self.labels = labels

    def __getitem__(self, key):
        """ Get an element/slice from the underlying data and labels """
        index = key if self.index is None else self.index[key]
        return Data(self.collection, self.images[index], self.labels[index])

    def __iter__(self):
        """ Allow for unpacking like a tuple """
        return (self.images, self.labels).__iter__()

    def __len__(self):
        """ Return the length of the data """
        return len(self.images)

    def copy(self):
        """ Return a shallow copy of the data """
        return Data(self.collection, self.images, self.labels)

    def shuffle(self):
        """ Method to shuffle the data and labels """
        self.index = np.random.permutation(self.images.shape[0])


class Dataset(object):
    """ Class that holds all the data for training/validation/testing/stereotyping. """
    def __init__(self, test, train, validation):
        self.test = test
        self.train = train
        self.validation = validation

    @property
    def image_size(self):
        """ Return the image size of the images in the dataset. """
        return self.train.images.shape[1:3]

    @property
    def image_width(self):
        """ Return the width of the images in the dataset. """
        return self.image_size[1]

    @property
    def image_height(self):
        """ Return the height of the images in the dataset. """
        return self.image_size[0]

    @property
    def num_channels(self):
        """ Return the number of color channels of the images in the dataset. """
        return self.train.images.shape[3]

    @property
    def num_labels(self):
        """ Return the number labels for the images in the dataset. """
        return np.max(self.train.labels) + 1

    def copy(self):
        """ Return a shallow copy of the dataset. """
        return Dataset(self.test.copy(), self.train.copy(), self.validation.copy())


class DatasetFactory(object):
    """ Class that holds all the data for training/validation/testing. """
    def __init__(self, datadir):
        self.loaders = {}
        self.datadir = datadir

    def register(self, dataset, dataloader):
        """ Register a dataloader with the factory. """
        assert issubclass(dataloader, DataLoader)
        self.loaders[dataset] = dataloader(self.datadir)

    def load_data(self, dataset, validation_percent=0.1, **kwargs):
        """ Register a dataloader with the factory. """
        return self.loaders[dataset].load_data(validation_percent=validation_percent, **kwargs)
