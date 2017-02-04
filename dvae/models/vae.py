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

"""Variational autoencoder model for images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dvae.models.factory import Model
from dvae.models.factory import ModelFactoryFunction


class VariationalAutoencoderModelFactory(ModelFactoryFunction):
    """ Definition of the model """
    def __init__(self, dataset, dtype):
        super(VariationalAutoencoderModelFactory, self).__init__(dataset, dtype)

    def define_model(self, graph, samples=20, recognition=None, reuse=None, **kwargs):
        """
        Define a VariationalAutoencoderModel.

        For more details see Auto-Encoding Variational Bayes:
        https://arxiv.org/pdf/1312.6114v10.pdf

        Args:
            samples: The number of samples from the unit Gaussian
            recognition: Model to generate q(z|x). Required parameter.
            the model, but can be set later on the VariationalAutoencoderModel.
            reuse: Whether to reuse variables

        Returns:
            A VariationalAutoencoderModel
        """
        if recognition is None:
            raise TypeError('define_model() needs keyword only argument recognition')

        with tf.variable_scope('mean', reuse=reuse):
            mean = self.linear_layers(
                recognition.output_tensor, (samples), reuse=reuse)[-1]

        with tf.variable_scope('log_variance', reuse=reuse):
            log_variance = self.linear_layers(
                recognition.output_tensor, (samples), reuse=reuse)[-1]

        kl_divergence = tf.mul(0.5, tf.reduce_sum(
            tf.square(mean) + tf.exp(log_variance)
            - log_variance - 1, 1), name='kl_divergence')

        sampled = tf.random_normal(tf.shape(log_variance), 0, 1, dtype=self.dtype)
        prior = mean + tf.sqrt(tf.exp(log_variance)) * sampled

        return VariationalAutoencoderModel(graph, recognition, prior, kl_divergence)


class VariationalAutoencoderModel(Model):
    """ An instance of an image classification model. """
    def __init__(self, graph, recognition, prior, kl_divergence):
        super(VariationalAutoencoderModel, self).__init__(graph, None, {})

        self.kl_divergence = kl_divergence

        self.prior = prior
        self._generation = None
        self.recognition = recognition
        self._placeholders['inputs'] = recognition.inputs

    @property
    def evaluation(self):
        """ Return the evaluation output. """
        return self.output_tensor

    def get_collection(self, key):
        """ Get the elements in the collection scoped to this model """
        collection = set()
        collection.update(super(VariationalAutoencoderModel, self).get_collection(key))
        collection.update(self.generation.get_collection(key))
        collection.update(self.recognition.get_collection(key))

        return collection

    def _initialize_metrics(self, metrics):
        """ Intitialize the model metrics """
        super(VariationalAutoencoderModel, self)._initialize_metrics(metrics)

        self.recognition._initialize_metrics(metrics)
        self.generation._initialize_metrics(metrics)

        scalar, update_op = tf.contrib.metrics.streaming_mean(self.kl_divergence)
        self._add_metric(metrics, 'kl_divergence', {
            'format': '{:.3f}',
            'update_op': update_op,
            'scalar': scalar})

    def collect_metrics(self, collection, metrics):
        """ Compute any needed metrics to display while training """
        super(VariationalAutoencoderModel, self).collect_metrics(collection, metrics)

        self.recognition.collect_metrics(collection, metrics)
        self.generation.collect_metrics(collection, metrics)

    def _initialize_summaries(self, summaries):
        """ Intitialize the model summaries """
        super(VariationalAutoencoderModel, self)._initialize_summaries(summaries)

        self.recognition._initialize_summaries(summaries)
        self.generation._initialize_summaries(summaries)

    def collect_summaries(self, collection, summaries):
        """ Compute any needed summaries """
        super(VariationalAutoencoderModel, self).collect_summaries(collection, summaries)

        self.recognition.collect_summaries(collection, summaries)
        self.generation.collect_summaries(collection, summaries)

    def feed(self, feed_dict, data, training=False):
        """ Feed any additional placeholders based on the training state. """
        super(VariationalAutoencoderModel, self).feed(feed_dict, data, training=training)

        self.recognition.feed(feed_dict, data, training=training)
        self.generation.feed(feed_dict, data, training=training)

    @property
    def generation(self):
        """ Return the variational autoencoder's generation model. """
        return self._generation

    @generation.setter
    def generation(self, generation):
        if generation is None:
            raise ValueError('Must have a valid model for generation!')

        self._generation = generation
        self.output_tensor = generation.output_tensor
        self._placeholders['targets'] = generation.targets

        scope = self.variable_scope
        device = self.kl_divergence.device
        with self.graph.as_default(), tf.device(device), tf.variable_scope(scope, reuse=True):
            self._initialize_loss(self.kl_divergence)
