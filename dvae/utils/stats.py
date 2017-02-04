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

import contextlib

import six
import tensorflow as tf


_METRIC_SCOPE = 'metric/'
_SUMMARY_SCOPE = 'summary/'

@contextlib.contextmanager
def metric_scope(scope, graph=None):
    """ Returns a context manager for metric operations """
    if not isinstance(scope, six.string_types):
        raise TypeError('scope must be of a string type, not {0}'.format(type(scope)))

    if len(scope) == 0:
        raise ValueError('scope must not be an empty string')

    scope = _METRIC_SCOPE + scope.rstrip('/') + '/'
    graph = tf.get_default_graph() if graph is None else graph
    with graph.as_default(), tf.device('/cpu:0'), tf.name_scope(scope):
        yield scope


@contextlib.contextmanager
def summary_scope(scope, graph=None):
    """ Returns a context manager for metric operations """
    if not isinstance(scope, six.string_types):
        raise TypeError('scope must be of a string type, not {0}'.format(type(scope)))

    if len(scope) == 0:
        raise ValueError('scope must not be an empty string')

    scope = _SUMMARY_SCOPE + scope.rstrip('/') + '/'
    graph = tf.get_default_graph() if graph is None else graph
    with graph.as_default(), tf.device('/cpu:0'), tf.name_scope(scope):
        yield scope
