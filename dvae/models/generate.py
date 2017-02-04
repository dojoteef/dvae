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

"""Transpose convolutional model for image generation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import reduce  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.models.factory import Model
from dvae.models.factory import ModelFactoryFunction
import dvae.utils.image as image_utils


class GenerationModelFactory(ModelFactoryFunction):
    """ Definition of the model """
    def __init__(self, dataset, dtype):
        super(GenerationModelFactory, self).__init__(dataset, dtype)

    def _transpose_convolution(self, data, weight_shape, output_shape):
        """ Define a convolution layer. """
        dtype = self.dtype

        weights = tf.get_variable(
            "weights", dtype=dtype, shape=weight_shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))

        biases = tf.get_variable(
            "biases", dtype=dtype, shape=weight_shape[-2:-1],
            regularizer=tf.contrib.layers.l2_regularizer(5e-4),
            initializer=tf.zeros_initializer)

        conv_transpose = tf.nn.conv2d_transpose(
            data, weights, output_shape,
            strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.bias_add(conv_transpose, biases)

    def _transpose_convolutions(self, layer, filters, channels, reuse=None):
        image_size = layer.get_shape().as_list()[1:3]

        for idx in xrange(len(channels) - 1):
            with tf.variable_scope('trans_conv'+str(idx), reuse=reuse):
                shape = tf.shape(layer)
                image_size = [2 * x for x in image_size]
                output_shape = [shape[0], image_size[0], image_size[1], channels[idx+1]]
                weight_shape = filters + channels[idx+1:idx-1 if idx-1 >= 0 else None:-1]

                layer = self._transpose_convolution(layer, weight_shape, output_shape)
                if idx < len(channels) - 2:
                    layer = self.batch_norm(layer, activation_fn=tf.nn.relu, reuse=reuse)

        return layer

    def define_model(self, graph, channels=(64, 32), hidden=(512,), input_tensor_or_shape=None,
                     reuse=None, **kwargs):
        """ Return a new model. """
        if input_tensor_or_shape is None:
            raise TypeError('define_model() needs keyword only argument input_tensor_or_shape')

        placeholders = {}
        if isinstance(input_tensor_or_shape, tf.TensorShape):
            input_shape = input_tensor_or_shape
            inputs = tf.placeholder(self.dtype, name='inputs', shape=input_shape)

            placeholders['inputs'] = inputs
        elif isinstance(input_tensor_or_shape, tf.Tensor):
            inputs = input_tensor_or_shape
            input_shape = inputs.get_shape()
        else:
            raise TypeError('input_tensor_or_shape must be a tf.Tensor or tf.TensorShape')

        placeholders['targets'] = tf.placeholder(self.dtype, name='targets', shape=(
            None, self.dataset.image_height, self.dataset.image_width, self.dataset.num_channels))

        # Make sure whatever input is received is flattened
        # First dimension is the batch dimension, so no need to fully specify it
        if len(input_shape) > 2:
            collapsed = int(reduce(lambda n, m: n*m, input_shape[1:]))
            flattened = tf.reshape(inputs, [-1, collapsed])
        else:
            flattened = inputs

        # The final linear layer needs to output the correct size based on the number of channels
        # and what the initial image size should be based on the convolution stride
        image_size = self.dataset.image_size
        image_dimensions = (tf.Dimension(dim) for dim in image_size)
        dimension_reduction = tf.Dimension(pow(2, len(channels)))
        initial_image_size = tuple(int(dim // dimension_reduction) for dim in image_dimensions)

        initial_image_shape = [-1, initial_image_size[0], initial_image_size[1], channels[0]]
        flattened_image_size = int(reduce(lambda n, m: n*m, initial_image_shape[1:]))

        features = list(hidden)
        features.append(flattened_image_size)

        linear = self.linear_layers(
            flattened,
            features,
            activation_fn=tf.nn.relu,
            reuse=reuse)[-1]

        convolution_channels = list(channels)
        convolution_channels.append(self.dataset.num_channels)

        # Reshape the image to the initial shape
        # First dimension is the batch dimension, so no need to fully specify it
        image = tf.reshape(linear, initial_image_shape)

        image = self._transpose_convolutions(
            image,
            [5, 5],
            convolution_channels,
            reuse=reuse)

        return GenerationModel(graph, image, placeholders)


class GenerationModel(Model):
    """ An instance of an image classification model. """
    def __init__(self, graph, output_tensor, placeholders):
        generator = tf.nn.sigmoid(output_tensor)
        super(GenerationModel, self).__init__(graph, generator, placeholders)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(output_tensor, self.targets)
        generation_loss = tf.reduce_sum(cross_entropy, tf.range(1, tf.rank(cross_entropy)))
        self._initialize_loss(generation_loss)

    @property
    def evaluation(self):
        """ Return the evaluation output. """
        return self.output_tensor

    def _initialize_summaries(self, summaries):
        """ Initialize the model summaries """
        super(GenerationModel, self)._initialize_summaries(summaries)
        summaries.append(tf.summary.image(
            self.variable_scope,
            image_utils.tile_images(self.evaluation),
            max_outputs=1))

    def feed(self, feed_dict, data, training=False):
        """ Feed any additional placeholders based on the training state. """
        super(GenerationModel, self).feed(feed_dict, data, training=training)

        feed_dict[self.targets] = data.images
