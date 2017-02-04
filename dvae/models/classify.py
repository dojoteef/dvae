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

"""Convolutional model for image classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.models.factory import Model
from dvae.models.factory import ModelFactoryFunction
import dvae.utils.image as image_utils


class ClassificationModelFactory(ModelFactoryFunction):
    """ Definition of the model """
    def __init__(self, dataset, dtype):
        super(ClassificationModelFactory, self).__init__(dataset, dtype)

    def _convolution(self, data, shape, reuse=None):
        """ Define a convolution layer. """
        dtype = self.dtype

        weights = tf.get_variable(
            "weights", dtype=dtype, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))

        biases = tf.get_variable(
            "biases", dtype=dtype, shape=shape[-1:],
            regularizer=tf.contrib.layers.l2_regularizer(5e-4),
            initializer=tf.zeros_initializer)

        conv = tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')
        norm = self.batch_norm(tf.nn.bias_add(conv, biases), activation_fn=tf.nn.relu, reuse=reuse)
        return tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _convolutions(self, layer, filters, channels, reuse=None):
        for idx in xrange(len(channels) - 1):
            with tf.variable_scope('conv'+str(idx), reuse=reuse):
                layer = self._convolution(layer, filters + channels[idx:idx+2], reuse=reuse)

        return layer

    def define_model(self, graph, channels=(32, 64), hidden=(512,),
                     input_tensor=None, reuse=None, **kwargs):
        """ Return a new model. """
        placeholders = {}
        placeholders['inputs'] = tf.placeholder(self.dtype, name='inputs', shape=(
            None, self.dataset.image_height, self.dataset.image_width, self.dataset.num_channels))
        placeholders['targets'] = tf.placeholder(tf.int64, name='targets')

        if input_tensor is not None:
            if not isinstance(input_tensor, tf.Tensor):
                raise TypeError('input_tensor must be a tf.Tensor')
            inputs = input_tensor
        else:
            inputs = placeholders['inputs']

        convolution_channels = [self.dataset.num_channels]
        convolution_channels.extend(channels)

        filters = [5, 5]
        convolutions = self._convolutions(
            inputs,
            filters,
            convolution_channels,
            reuse=reuse)

        # Need to reshape the output of the convolutions to feed into the linear layers.
        reduction = 2**(len(convolution_channels) - 1)
        conv_width = math.ceil(self.dataset.image_width / reduction)
        conv_height = math.ceil(self.dataset.image_height / reduction)
        collapsed = convolution_channels[-1] * conv_width * conv_height

        # First dimension is the batch dimension, so no need to fully specify it
        flattened = tf.reshape(convolutions, [-1, collapsed])

        features = list(hidden)
        features.append(self.dataset.num_labels)

        # Use dropout for normalization during training
        layers = self.linear_layers(
            flattened,
            features,
            activation_fn=tf.nn.relu,
            reuse=reuse)

        return ClassificationModel(graph, layers[-2:], placeholders, input_tensor=input_tensor)


class ClassificationModel(Model):
    """ An instance of an image classification model. """
    def __init__(self, graph, outputs, placeholders, input_tensor=None):
        super(ClassificationModel, self).__init__(graph, outputs[0], placeholders)

        self._input_tensor = input_tensor
        self._classifier = tf.argmax(tf.nn.softmax(outputs[1]), 1)

        correct = tf.equal(self.evaluation, self.targets)
        self.error_rate = 1.0 - tf.reduce_mean(tf.cast(correct, tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs[1], labels=self.targets)

        self._initialize_loss(loss)

    @property
    def evaluation(self):
        """ Return the evaluation output. """
        return self._classifier

    @property
    def inputs(self):
        """ Return the inputs to the classifier """
        if self._input_tensor is not None:
            return self._input_tensor

        return super(ClassificationModel, self).inputs

    def _initialize_metrics(self, metrics):
        """ Initialize model metrics """
        super(ClassificationModel, self)._initialize_metrics(metrics)
        scalar, update_op = tf.contrib.metrics.streaming_accuracy(
            self.evaluation, self.targets)
        self._add_metric(metrics, 'error_rate', {
            'format': '{:.2%}',
            'update_op': update_op,
            'scalar': 1.0 - scalar})

    def _initialize_summaries(self, summaries):
        """ Initialize the model summaries """
        super(ClassificationModel, self)._initialize_summaries(summaries)
        summaries.append(tf.summary.image(
            self.variable_scope,
            image_utils.tile_images(self.inputs),
            max_outputs=1))

    def feed(self, feed_dict, data, training=False):
        """ Feed any additional placeholders based on the training state. """
        super(ClassificationModel, self).feed(feed_dict, data, training=training)

        if self.inputs.op.type == 'Placeholder':
            feed_dict[self.inputs] = data.images

        feed_dict[self.targets] = data.labels
