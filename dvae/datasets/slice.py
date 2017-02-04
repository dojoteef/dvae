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

""" A dataloader that divvies up an existing dataset """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from dvae.datasets.dataloader import Data
from dvae.datasets.dataloader import Dataset
from dvae.datasets.dataloader import DataLoader


class SliceDataLoader(DataLoader):
    """ Class for manipulating the data. """
    def __init__(self, datadir):
        super(SliceDataLoader, self).__init__(datadir)

    @property
    def image_count(self):
        """ Return the number of images in the dataset. """
        return 60000

    @property
    def test_image_count(self):
        """ Return the number of test images in the dataset. """
        return 10000

    @property
    def image_size(self):
        """ Return the image size of the images in the dataset. """
        return 28

    @property
    def num_channels(self):
        """ Return the number of color channels of the images in the dataset. """
        return 1

    @property
    def num_labels(self):
        """ Return the number labels for the images in the dataset. """
        return 10

    def slice_data(self, data, slice_percent):
        """ Return a slice of the following data. """
        images = data.images
        labels = data.labels
        count = int(slice_percent * len(labels))
        index = np.random.permutation(len(labels))

        indexed_slice = index[:count, ...]
        data_slice = Data(data.collection, images[indexed_slice], labels[indexed_slice])

        remaining_slice = index[count:, ...]
        images = images[remaining_slice]
        labels = labels[remaining_slice]

        return data_slice

    def load_data(self, data=None, slice_percent=0.0, **kwargs):
        """ Load the data used for training/validation/testing."""
        if data is None:
            raise ValueError('Must provide data to slice')

        train = self.slice_data(data.train, slice_percent)
        validation = self.slice_data(data.validation, slice_percent)
        test = self.slice_data(data.test, slice_percent)

        return Dataset(test, train, validation)
