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

""" Utilities for calculating Tensorflow statistics """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    """ Get the number of available gpus """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

_INVALID_DEVICE_CHARACTERS = re.compile(r'[^\w]')
def _training_variable_from_device(device):
    """ Return the name of the training variable given the device """
    device_name = _INVALID_DEVICE_CHARACTERS.sub('_', device).strip('_')
    return 'is_training_{0}'.format(device_name)

def trainable_graph(num_gpus):
    """ Return a trainable graph with an is_training placeholder """
    graph = tf.Graph()
    with graph.as_default():
        devices = ['/cpu:0'] + ['/gpu:{0}'.format(gpu_index) for gpu_index in xrange(num_gpus)]
        for device in devices:
            name = _training_variable_from_device(device)
            training_variable = tf.get_variable(
                name, shape=[], dtype=tf.bool,
                initializer=tf.constant_initializer(False),
                trainable=False, caching_device=device)
            graph.add_to_collection('IS_TRAINING', training_variable)

    return graph

def initialize_training(graph=None):
    """ Initialize the is_training variables """
    graph = tf.get_default_graph() if graph is None else graph

    initialization_ops = []
    training_variables = graph.get_collection('IS_TRAINING')
    for training_variable in training_variables:
        initialization_ops.append(training_variable.initializer)

    return tf.group(*initialization_ops)

def is_training(node):
    """ Get the is_training variable for the specified node """
    device = node.device
    if device is None or len(device) == 0:
        raise ValueError(
            'node.device must be specified, \
            scope using tf.device() when creating the node')

    graph = tf.get_default_graph() if node.graph is None else node.graph
    training_variables = graph.get_collection('IS_TRAINING')
    training_variable_name = '{0}:0'.format(_training_variable_from_device(device))

    for training_variable in training_variables:
        if training_variable.name == training_variable_name:
            return training_variable

def set_training(training, graph=None, session=None):
    """ Set the graph into training mode """
    session = tf.get_default_session() if session is None else session
    graph = tf.get_default_graph() if graph is None else graph

    training_ops = []
    training_variables = graph.get_collection('IS_TRAINING')
    for training_variable in training_variables:
        training_ops.append(training_variable.assign(training))

    return tf.group(*training_ops)

# Reused scopes have an '_<digits>' for the reuse count which will count as a match when
# calling the scope_index function
_SCOPE_MATCH = re.compile(r'([a-zA-Z]+)(_\d+)?')
def scope_index(full_scope, subscope):
    """ Get the index of the passed in scope from the parent scope """
    scopes = [_SCOPE_MATCH.match(scope).group(1) for scope in full_scope.split('/')]
    subscopes = subscope.split('/')
    count = len(subscopes)

    for index in (index for index, element in enumerate(scopes) if element == subscopes[0]):
        if scopes[index:index+count] == subscopes:
            return index

    return None
