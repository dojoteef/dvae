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

"""Base class for the model factory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
from abc import abstractproperty
from abc import ABCMeta as AbstractBaseClass

from six import iteritems
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.datasets.dataloader import Data
import dvae.utils.graph as graph_utils
import dvae.utils.stats as stats_utils


# Collection key for batch norm variables
BATCH_NORM = 'BATCH_NORM'
BATCH_NORM_COLLECTION = {'moving_mean': [BATCH_NORM], 'moving_variance': [BATCH_NORM]}


class ModelFactoryFunction(object):
    """ Base class for individual model factory creation functions. """
    __metaclass__ = AbstractBaseClass

    def __init__(self, dataset, dtype):
        self.dtype = dtype
        self.dataset = dataset

    def batch_norm(self, layer, activation_fn=None, reuse=None):
        """ Unified batch norm method for all models """
        batch_norm_layer = tf.contrib.layers.batch_norm(
            layer, decay=0.999, trainable=False,
            is_training=graph_utils.is_training(layer),
            scale=True, center=True, activation_fn=activation_fn,
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            variables_collections=BATCH_NORM_COLLECTION,
            scope='batch_norm', reuse=reuse)

        return batch_norm_layer

    def linear_layer(self, data, shape):
        """ Create a single linear layer with the specified shape """
        dtype = self.dtype

        weights = tf.get_variable(
            'weights', dtype=dtype, shape=shape,
            regularizer=tf.contrib.layers.l2_regularizer(5e-4),
            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        biases = tf.get_variable(
            'biases', dtype=dtype, shape=shape[-1:],
            regularizer=tf.contrib.layers.l2_regularizer(5e-4),
            initializer=tf.zeros_initializer)

        return tf.matmul(data, weights) + biases

    def linear_layers(self, layer, outputs, activation_fn=None, reuse=None, name='linear'):
        """ Create a linear layer for each entry in outputs """
        layers = []
        outputs = layer.get_shape()[-1:].concatenate(outputs)
        for idx in xrange(len(outputs) - 1):
            with tf.variable_scope(name+str(idx), reuse=reuse):
                activate = activation_fn if idx < len(outputs) - 2 else None

                layer = self.linear_layer(layer, outputs[idx:idx+2])
                if activation_fn:
                    layer = self.batch_norm(layer, activation_fn=activate, reuse=reuse)
                layers.append(layer)

        return layers

    @abstractmethod
    def define_model(self, graph, reuse=None, **kwargs):
        """ Return a new model. """
        pass


class Model(object):
    """ An instance of an image classification model. """
    __metaclass__ = AbstractBaseClass

    def __init__(self, graph, output_tensor, placeholders):
        self.graph = graph

        self._loss = None
        self._placeholders = placeholders
        self.metrics = {Data.TRAIN: {}, Data.VALIDATE: {}, Data.TEST: {}}
        self.summaries = {Data.TRAIN: [], Data.VALIDATE: [], Data.TEST: []}

        self.name = None
        self._output_tensor = None
        self._variable_scope = tf.get_variable_scope()

        self.output_tensor = output_tensor

    @property
    def output_tensor(self):
        """ Return the type of the activations, weights, and placeholder variables. """
        return self._output_tensor

    @output_tensor.setter
    def output_tensor(self, output_tensor):
        """ Return the type of the activations, weights, and placeholder variables. """
        # There is no easy way to find the name scope, so just take the name of the output tensor
        # (which will have the full scope) and use the portion before the variable scope
        if output_tensor is not None:
            scopes = output_tensor.name.split('/')
            scope_index = graph_utils.scope_index(output_tensor.name, self.variable_scope)
            self.name = '/'.join(scopes[:scope_index])

        self._output_tensor = output_tensor

    @property
    def dtype(self):
        """ Return the type of the activations, weights, and placeholder variables. """
        return self.output_tensor.dtype

    @property
    def inputs(self):
        """ Get the placeholder for model inputs """
        return self._placeholders['inputs']

    @property
    def targets(self):
        """ Get the placeholder for model targets """
        return self._placeholders['targets']

    @property
    def full_name(self):
        """ Return the full name of the model (name scope + variable scope). """
        scopes = [] if self.name is None else [self.name]
        scopes.append(self.variable_scope)

        return '/'.join(scopes)

    @property
    def variable_scope(self):
        """ Return the variable scope of the model. """
        return self._variable_scope.name

    @property
    def global_variables(self):
        """ Get all the variables of the model """
        return self.get_collection(tf.GraphKeys.VARIABLES)

    @property
    def trainable_variables(self):
        """ Get the trainable variables of the model """
        return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def update_ops(self):
        """ Get the update operations of the model """
        return self.get_collection(tf.GraphKeys.UPDATE_OPS)

    @property
    def loss(self):
        """ Return the model loss. """
        return self._loss

    def _initialize_loss(self, loss):
        """ Initialize the model loss """
        if loss is None:
            raise ValueError('Loss must not be None!')

        losses = tf.contrib.losses.get_regularization_losses(scope=self.variable_scope)
        if len(losses) > 0:
            loss = tf.add(loss, tf.add_n(losses))

        self._loss = loss

    @abstractproperty
    def evaluation(self):
        """ Return the evaluation output. """
        return None

    @abstractmethod
    def feed(self, feed_dict, data, training=False):
        """ Feed any additional placeholders based on the training state. """
        pass

    def initialize_metrics(self):
        """ Initialize model metrics """
        if not self._variable_scope.reuse:
            for scope in (Data.TRAIN, Data.VALIDATE, Data.TEST):
                with stats_utils.metric_scope(scope, graph=self.graph):
                    self._initialize_metrics(self.metrics[scope])

    def _add_metric(self, metrics, name, value):
        """ Add a metric to the collection of metrics """
        metric_name = '{0}/{1}'.format(self.variable_scope, name)
        if metric_name in metrics:
            raise ValueError('Duplicate metric defined!')

        metrics[metric_name] = value

    def _initialize_metrics(self, metrics):
        """ Initialize the model metrics """
        scalar, update_op = tf.contrib.metrics.streaming_mean(self.loss)
        self._add_metric(metrics, 'loss', {
            'format': '{:.3f}',
            'update_op': update_op,
            'scalar': scalar})

    def initialize_summaries(self):
        """ Initialize model summaries """
        if not self._variable_scope.reuse:
            for scope in (Data.TRAIN, Data.VALIDATE, Data.TEST):
                with stats_utils.summary_scope(scope, graph=self.graph):
                    summaries = self.summaries[scope]
                    self._initialize_summaries(summaries)

                    for (name, metric) in iteritems(self.metrics[scope]):
                        summaries.append(tf.summary.scalar(name, metric['scalar']))

                    if scope == Data.TRAIN:
                        for variable in self.get_collection(BATCH_NORM):
                            summaries.append(tf.summary.histogram(variable.op.name, variable))
                            summaries.append(tf.summary.scalar(
                                variable.op.name + '_avg',
                                tf.reduce_mean(variable)))

    def _initialize_summaries(self, summaries):
        """ Initialize the model metrics """
        pass

    def collect_metrics(self, collection, metrics):
        """ Compute any needed metrics to display while training """
        metrics.update(self.metrics[collection])

    def collect_summaries(self, collection, summaries):
        """ Compute any needed summaries """
        summaries.extend(self.summaries[collection])

    def get_collection(self, key):
        """ Get the elements in the collection scoped to this model """
        collection = []
        for scope in (self.variable_scope, self.full_name):
            collection.extend(self.graph.get_collection(key, scope=scope))

        return collection


class ModelFactory(object):
    """ Factory for creating models. """
    def __init__(self, dataset):
        self.dataset = dataset
        self.factories = {}

    @property
    def dtype(self):
        """ Return the type of the activations, weights, and placeholder variables. """
        return tf.float32

    def register(self, model_type, model_function):
        """ Register a model factory function. """
        assert issubclass(model_function, ModelFactoryFunction)
        self.factories[model_type] = model_function(self.dataset, self.dtype)

    def define_model(self, model_type,
                     graph=None, scope=None,
                     device=None, reuse=None, **kwargs):
        """ Return a new model. """
        factory = self.factories[model_type]

        graph = tf.Graph() if graph is None else graph
        device = '/cpu:0' if device is None else device

        with graph.as_default(), tf.variable_scope(scope, reuse=reuse), tf.device(device):
            model = factory.define_model(graph, reuse=reuse, **kwargs)

        return model
