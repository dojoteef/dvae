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

"""Main entry point for training/testing of the stereotyping model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import argparse

import six

import dvae.utils.graph as graph_utils


def _json_parser():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        action='store',
        help='How many images per batches.'
    )
    parser.add_argument(
        '--denoising',
        default=False,
        action='store_true',
        help='Whether to make the variational autoencoder denoising or not'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        action='store',
        help='Initial learning rate'
    )
    parser.add_argument(
        '--decay_step',
        type=float,
        default=25,
        action='store',
        help='How many epochs for a single decay step'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        action='store',
        help='How many epochs for training.'
    )
    parser.add_argument(
        '--metric_frequency',
        type=int,
        default=10,
        action='store',
        help='The number of steps between updating metrics, 0 disable metrics'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=20,
        action='store',
        help='The size of the samples from the latent space'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        action='store',
        help='The number of samples from the latent space'
    )
    parser.add_argument(
        '--trace_frequency',
        type=int,
        default=0,
        action='store',
        help='The number of steps between running trace operations, 0 disables traces'
    )
    parser.add_argument(
        '--validation_frequency',
        type=int,
        default=1000,
        action='store',
        help='The number of steps between validation, 0 disables validation'
    )

    return parser

def _parseable(pair):
    """ Return whether the pair is parseable """
    string_type = isinstance(pair[1], six.string_types)
    is_namespace = isinstance(pair[1], argparse.Namespace)
    iterable_type = isinstance(pair[1], collections.Iterable)
    return (string_type or not iterable_type) and not is_namespace

def _parser(defaults=None):
    """ Return a parser function """
    parser = _json_parser()

    if defaults:
        parser.set_defaults(**defaults)

    def parse(pairs):
        """ Validate the passed in json element """
        arg_list = [
            str(item)
            for pair in pairs
            for item in ('--' + pair[0], pair[1])
            if _parseable(pair)]

        parsed = {
            pair[0]: pair[1]
            for pair in pairs
            if not _parseable(pair)}

        if parsed and arg_list:
            raise ValueError('Nested configurations are not allowed')

        return parser.parse_args(arg_list) if arg_list else parsed

    return parse

def override(namespace, overrides):
    """ Override values in namespace with the passed in overrides """
    namespace = vars(namespace)
    namespace.update(overrides)

    return argparse.Namespace(**namespace)

def _overrides(arguments):
    """ Get a dict of overrides """
    parsed = vars(_json_parser().parse_args(arguments))
    arguments = [arguments[i].lstrip('-') for i in range(0, len(arguments), 2)]

    return {argument: parsed[argument] for argument in arguments}

def _load_file(filename, defaults=None):
    """ Load the configuration from a file """
    try:
        stream = open(filename, 'r')
    except Exception:
        raise ValueError('Cannot find config file: {0}'.format(filename))

    return json.load(stream, object_pairs_hook=_parser(defaults=defaults))

def load(arguments, config_file=None, defaults_file=None):
    """ Load a json config """
    defaults = _load_file(defaults_file) if defaults_file else {}
    config = _load_file(config_file, defaults=vars(defaults)) if config_file else {}

    if isinstance(config, dict):
        overrides = _overrides(arguments)
        for (key, namespace) in six.iteritems(config):
            config[key] = override(namespace, overrides)

    return config

def parse_commandline():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        action='store',
        help='JSON config file for the arguments'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='data',
        action='store',
        help='Where to store the data.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['cifar10', 'mnist'],
        action='store',
        help='Which dataset to use.'
    )
    parser.add_argument(
        '--defaults_file',
        type=str,
        default='',
        action='store',
        help='JSON config file for the argument defaults'
    )
    parser.add_argument(
        '--log_device_placement',
        default=False,
        action='store_true',
        help='Whether to log the device placement to stdout'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=len(graph_utils.get_available_gpus()),
        action='store',
        help='How many gpus to use for training.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        action='store',
        help='What random seed to use.'
    )
    parser.add_argument(
        '--validation_percent',
        type=float,
        default=0.1,
        action='store',
        help='What percentage of images to use for validation'
    )
    parser.add_argument(
        '--traindir',
        type=str,
        default='train',
        action='store',
        help='The directory to store training information into.'
    )

    return parser.parse_known_args()
