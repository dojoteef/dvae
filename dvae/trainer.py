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

"""The trainer for the convolutional model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
from six import iteritems  # pylint: disable=redefined-builtin
from six import itervalues  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.datasets.dataloader import Data
from dvae.models.factory import Model
import dvae.utils.graph as graph_utils
import dvae.utils.stats as stats_utils


class Tower(object):
    """ A single tower to be trained on a device. """
    def __init__(self, scope, device, models, loss, dataset):
        self.scope = scope
        self.device = device
        self.models = models
        self.dataset = dataset.copy()

        if not isinstance(models, (list, tuple, Model)):
            raise TypeError('models must either be a list, tuple, dvae.factory.Model')

        if isinstance(models, Model):
            models = (models)

        if len(models) == 0:
            raise ValueError('At least one model required for training')

        self.graph = models[0].graph
        for model in models[1:]:
            if self.graph != model.graph:
                raise KeyError('All models must be from the same graph!')

        self._initialize_metrics()
        self._initialize_summaries()
        self._initialize_loss(loss)

    def _initialize_metrics(self):
        """ Initialize the model metrics """
        for model in self.models:
            model.initialize_metrics()

    def _initialize_summaries(self):
        """ Initialize the model summaries """
        for model in self.models:
            model.initialize_summaries()

    def _initialize_loss(self, loss):
        """ Initialize the tower loss """
        update_ops = self.update_ops
        if len(update_ops) > 0:
            with tf.control_dependencies(update_ops):
                self.loss = tf.identity(loss)

    def get_collection(self, key):
        """ Get all the variables of the models in the tower """
        collection = set()
        for model in self.models:
            collection.update(model.get_collection(key))

        return collection

    @property
    def global_variables(self):
        """ Get all the variables of the models in the tower """
        return self.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    @property
    def trainable_variables(self):
        """ Get the trainable variables of the models in the tower """
        return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def update_ops(self):
        """ Get the update operations of the models in the tower """
        return self.get_collection(tf.GraphKeys.UPDATE_OPS)

    def compute_gradients(self, optimizer):
        """ Compute the gradients for the model """
        var_list = set()
        for model in self.models:
            var_list.update(model.trainable_variables)

        with tf.name_scope(self.scope), tf.device(self.device):
            kwargs = {'var_list': var_list}
            grads_and_vars = optimizer.compute_gradients(self.loss, **kwargs)

            with tf.name_scope('clip_gradients'):
                return [
                    (tf.clip_by_value(grad, -5, 5), var)
                    for grad, var in grads_and_vars
                    if grad is not None]

    def feed(self, feed_dict, data, training=False):
        """ Feed all the models in the tower """
        for model in self.models:
            model.feed(feed_dict, data, training=training)

    def collect_summaries(self, collection, summaries):
        """ Collect summaries for all the models in the tower """
        for model in self.models:
            model.collect_summaries(collection, summaries)

    def collect_metrics(self, collection, metrics):
        """ Collect the metrics from the models for the values being fetched """
        for model in self.models:
            model.collect_metrics(collection, metrics)


class ModelTrainer(object):
    """ Class used to train a model. """
    def __init__(self, session, batch_size, towers,
                 decay_step, learning_rate, summary_writer, name=None):
        if not isinstance(towers, (list, tuple, Tower)):
            raise TypeError('towers must either be a list, tuple, dvae.trainer.Tower')

        if isinstance(towers, Model):
            towers = (towers)

        if len(towers) == 0:
            raise ValueError('At least one tower required for training')

        self.data = towers[0].dataset
        self.graph = towers[0].graph
        for tower in towers[1:]:
            if self.graph != tower.graph:
                raise KeyError('All towers must be from the same graph!')

        self.batch_size = batch_size
        self.name = name
        self.session = session
        self.summary_writer = summary_writer
        self.timer = ((-1, 0), (-1, 0))
        self.towers = towers

        with self.graph.as_default(), tf.variable_scope(self.name, default_name='trainer'):
            self._initialize_metrics()
            self._init_training(decay_step, learning_rate)
            self._init_variables()

            with stats_utils.summary_scope('training', graph=self.graph):
                self._init_summaries()

    @property
    def train_dir(self):
        """ Get the training directory """
        return self.summary_writer.get_logdir()

    @property
    def global_step(self):
        """ Get the current global step value """
        return tf.train.global_step(self.session, self.global_step_tensor)

    @property
    def training_samples(self):
        """ Get the total number of training samples """
        return len(self.data.train)

    def _init_training(self, decay_step, learning_rate):
        """ Initialization of the training parameters """
        self.global_step_tensor = tf.Variable(
            tf.constant(0, tf.int32, shape=[]), False,
            name='global_step', collections=[tf.GraphKeys.LOCAL_VARIABLES])

        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step_tensor * self.batch_size,
            int(decay_step * self.training_samples), 0.1, staircase=True), 1e-5)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.gradients = []
        with tf.name_scope('compute_gradients'):
            for grads in zip(*[tower.compute_gradients(self.optimizer) for tower in self.towers]):
                gradient = tf.reduce_mean(tf.stack([grad for grad, _ in grads]), 0)
                self.gradients.append((gradient, grads[0][1]))

        enable_training = graph_utils.set_training(True, graph=self.graph, session=self.session)
        disable_training = graph_utils.set_training(False, graph=self.graph, session=self.session)

        self.summary_operation = disable_training

        self.evaluation_operations = {}
        for data_scope in (Data.VALIDATE, Data.TEST):
            self.evaluation_operations[data_scope] = tf.group(
                disable_training, *self.update_metrics[data_scope])

        with tf.control_dependencies([enable_training]):
            update_ops = self._update_ops()
            with tf.control_dependencies([update_ops]):
                apply_gradients = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step_tensor)

                self.training_operation = tf.group(
                    enable_training, update_ops,
                    self.learning_rate, apply_gradients,
                    *self.update_metrics[Data.TRAIN])

    def _update_ops(self):
        """ Get all the update ops for the towers """
        update_ops = []
        for tower in self.towers:
            update_ops.extend(tower.update_ops)

        return tf.group(*update_ops)

    def _initialize_metrics(self):
        """ Initialize the model metrics """
        self.metrics = {}
        self.metric_values = {}
        self.update_metrics = {}
        self.reset_metrics = {}
        for data_scope in (Data.TRAIN, Data.VALIDATE, Data.TEST):
            metrics = self.collect_metrics(data_scope)
            self.metrics[data_scope] = metrics

            self.metric_values[data_scope] = {
                name: metric['scalar']
                for name, metric in iteritems(metrics)}

            self.update_metrics[data_scope] = [
                metric['update_op']
                for metric in itervalues(metrics)]

            metric_variables = []
            with stats_utils.metric_scope(data_scope, graph=self.graph) as scope:
                for local in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope):
                    metric_variables.append(local)
            self.reset_metrics[data_scope] = tf.variables_initializer(metric_variables)

    def _init_variables(self):
        """ Create the initialization operation for the variables """
        # Adam optimizer uses two variables that can only be accessed through the use of a protected
        # function since the variables aren't scoped in anyway. Trying to add a tf.variable_scope
        # around apply_gradients where the variables are created did not help.
        var_list = set(self.optimizer._get_beta_accumulators()) # pylint: disable=protected-access
        slot_names = self.optimizer.get_slot_names()
        for tower in self.towers:
            variables = tower.global_variables
            var_list.update(variables)

            for slot_name in slot_names:
                for variable in variables:
                    slot = self.optimizer.get_slot(variable, slot_name)
                    if slot is not None:
                        var_list.add(slot)

        # Initialize all the variables
        self.initialization_operation = tf.group(
            tf.variables_initializer(var_list),

            # Apparently local variables are not part of 'all' variables... go figure
            # This is needed for metrics for example
            tf.local_variables_initializer())

    def _init_summaries(self):
        """ Initialization of the training parameters """
        summaries = []
        summaries.append(tf.summary.scalar('learning_rate', self.learning_rate))

        for gradient, variable in self.gradients:
            if gradient is not None:
                summaries.append(tf.summary.scalar(
                    variable.op.name + '/gradient_avg',
                    tf.reduce_mean(gradient)))

                summaries.append(tf.summary.histogram(
                    variable.op.name + '/gradients', gradient))

        for variable in tf.trainable_variables():
            summaries.append(tf.summary.histogram(variable.op.name, variable))

        self.training_summary = tf.summary.merge(summaries)

    def write_summaries(self, data):
        """ Write the summaries for the current step and collection """
        summaries = [self.summary_operation]
        if data.collection == Data.TRAIN:
            summaries.append(self.training_summary)

        feed_dict = {}
        for tower in self.towers:
            tower.feed(feed_dict, data)
            tower.collect_summaries(data.collection, summaries)

        for summary in summaries:
            self.summary_writer.add_summary(self.session.run(
                summary, feed_dict=feed_dict), global_step=self.global_step)

    def collect_metrics(self, collection):
        """ Collect the metrics for the values being fetched """
        metrics = {}
        for tower in self.towers:
            tower.collect_metrics(collection, metrics)

        return metrics

    def extract_metrics(self, metrics, values):
        """ Extract the metrics from the passed in values """
        for key, value in iteritems(values):
            if key in metrics:
                metrics[key]['value'] = value

        return metrics

    def run(self, operation, data, run_options=None, run_metadata=None):
        """ Execute a single batch for the given data """
        training = (data.collection == Data.TRAIN)

        feed_dict = {}
        for tower in self.towers:
            tower.feed(feed_dict, data, training=training)

        values = self.session.run(
            operation, feed_dict=feed_dict,
            options=run_options, run_metadata=run_metadata)

        metric_values = self.session.run(self.metric_values[data.collection], feed_dict=feed_dict)
        return values, self.extract_metrics(self.metrics[data.collection], metric_values)

    def evaluate(self, data):
        """ Evaluate the model with the given data. """
        # Reset any metrics before evaluating
        if data.collection not in self.evaluation_operations:
            raise ValueError('Cannot evaluate data from {0}'.format(data.collection))

        self.session.run(self.reset_metrics[data.collection])
        for step in xrange(int(np.ceil(len(data) / self.batch_size))):
            offset = (step * self.batch_size)
            batch = data[offset:(offset + self.batch_size), ...]
            _, metrics = self.run(self.evaluation_operations[data.collection], batch)

        self.write_summaries(data[:self.batch_size, ...])
        self.output_metrics(data, metrics)

    def optimize(self, data, with_metrics=False, with_trace=False):
        """ Optimize a single batch """
        run_metadata = tf.RunMetadata() if with_trace else None
        trace = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if with_trace else None

        _, metrics = self.run(
            self.training_operation, data,
            run_options=trace, run_metadata=run_metadata)

        if with_metrics:
            self.timer_update()
            steps, elapsed = self.elapsed()
            num_devices = len(self.towers)
            examples = steps * self.batch_size * num_devices
            print('Step {}, examples/sec {:.3f}, ms/batch {:.1f}'.format(
                self.global_step, examples / elapsed, 1000 * elapsed / num_devices))

            self.output_metrics(data, metrics)
            self.write_summaries(data)

        if with_trace:
            step = '{}/step{}'.format(self.name, self.global_step)
            self.summary_writer.add_run_metadata(run_metadata, step, global_step=self.global_step)

    def reset_timer(self):
        """ Reset the timer """
        self.timer = ((self.global_step, time.time()), (-1, 0))

    def timer_update(self):
        """ Update the current training timer """
        if self.timer[0][0] == self.global_step:
            return

        self.timer = ((self.global_step, time.time()), self.timer[0])

    def elapsed(self):
        """ Return the elapsed steps and time since the last training update """
        return tuple(np.subtract(*zip(self.timer)).squeeze())

    def output_metrics(self, data, metrics):
        """ Output the current training metrics """
        print('{} {}'.format(data.collection, ', '.join(
            [('{}: ' + metric['format']).format(name, metric['value'])
             for name, metric in iteritems(metrics)])))
        sys.stdout.flush()

    def shuffle(self):
        """ Shuffle the data in each tower """
        for tower in self.towers:
            tower.dataset.train.shuffle()

    def train(self, num_epochs, metric_frequency=0, validation_frequency=0, trace_frequency=0):
        """Generate a classification prediction using the passed in data """
        self.session.run(self.initialization_operation)
        self.reset_timer()

        # TODO: Have num_epochs account for training with multiple GPUs
        for _ in xrange(num_epochs):
            self.shuffle()

            for step in xrange(int(np.ceil(self.training_samples / self.batch_size))):
                offset = (step * self.batch_size)
                data = self.data.train[offset:(offset + self.batch_size), ...]

                # Global step is initialized to zero and isn't updated until a call to optimize
                # so need to add 1 for the current step being processed
                global_step = self.global_step + 1

                with_trace = trace_frequency > 0 and global_step % trace_frequency == 0
                with_metrics = metric_frequency > 0 and global_step % metric_frequency == 0
                self.optimize(data, with_metrics=with_metrics, with_trace=with_trace)

                if validation_frequency > 0 and global_step % validation_frequency == 0:
                    self.evaluate(self.data.validation)

        self.evaluate(self.data.test)
