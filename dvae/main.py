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

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

from dvae.datasets.cifar10 import CIFAR10DataLoader
from dvae.datasets.mnist import MNISTDataLoader
from dvae.datasets.slice import SliceDataLoader
from dvae.datasets.dataloader import DatasetFactory
from dvae.models.factory import ModelFactory
from dvae.models.classify import ClassificationModelFactory
from dvae.models.generate import GenerationModelFactory
from dvae.models.vae import VariationalAutoencoderModelFactory
from dvae.trainer import ModelTrainer
from dvae.trainer import Tower
import dvae.utils.config as config_utils
import dvae.utils.graph as graph_utils


def generate_classifier(model_factory, model_kwargs, config): # pylint: disable=unused-argument
    """ Generate the towers for the GPUs """
    classifier = model_factory.define_model(
        'classification', scope='classifier',
        channels=(128, 64, 32,), hidden=(1024, 512, 512,),
        **model_kwargs)

    return (classifier,), classifier.loss

def generate_vae(model_factory, model_kwargs, config):
    """ Generate the towers for the GPUs """
    scope_prefix = 'dvae' if config.denoising else 'vae'
    hidden = (512, 256,) if FLAGS.dataset == 'cifar10' else (384, 192,)
    channels = (128, 64,) if FLAGS.dataset == 'cifar10' else (64, 64,)

    vae_recognition = model_factory.define_model(
        'classification', scope='{0}/recognition'.format(scope_prefix),
        channels=channels, hidden=hidden,
        **model_kwargs)

    vae = model_factory.define_model(
        'vae', scope=scope_prefix,
        sample_size=config.sample_size,
        samples=config.samples,
        recognition=vae_recognition,
        **model_kwargs)

    vae_generation = model_factory.define_model(
        'generation', scope='{0}/generator'.format(scope_prefix),
        input_tensor_or_shape=vae.posterior,
        channels=channels, hidden=hidden[::-1],
        **model_kwargs)

    losses = []
    models = [vae]
    if config.denoising:
        recognition_kwargs = model_kwargs.copy()
        recognition_kwargs['reuse'] = True
        dvae_recognition = model_factory.define_model(
            'classification', scope='{0}/recognition'.format(scope_prefix),
            input_tensor=vae_generation.output_tensor,
            channels=channels, hidden=hidden,
            **recognition_kwargs)
        models.append(dvae_recognition)
        losses.append(dvae_recognition.loss)

        vae.generation = model_factory.define_model(
            'generation', scope='{0}/denoising/generator'.format(scope_prefix),
            input_tensor_or_shape=dvae_recognition.output_tensor,
            channels=channels, hidden=hidden[::-1],
            **model_kwargs)

        models.append(vae_generation)
        losses.append(vae_generation.loss)
    else:
        vae.generation = vae_generation

    losses.extend([vae.loss, vae.generation.loss, vae_recognition.loss])
    return models, tf.reduce_sum(losses)

_CPU_OPERATIONS = set(('Variable', 'VariableV2', 'Placeholder'))
def device_fn(device):
    """ Returns a function that given a tf.Operation it returns what device to put it on """
    def function(operation):
        """ Given a tf.Operation returns what device to put it on """
        if operation.type in _CPU_OPERATIONS:
            return '/cpu:0'
        else:
            return device

    return function

def generate_towers(graph, model_factory, dataset, model_function, config):
    """ Generate the towers for the GPUs """
    towers = []
    reuse = None

    gpus = graph_utils.get_available_gpus()
    for gpu_index in xrange(FLAGS.num_gpus):
        gpu = gpus[gpu_index]

        with graph.as_default(), tf.name_scope('tower{0}/'.format(gpu_index)) as name_scope:
            model_kwargs = {'graph': graph, 'device': device_fn(gpu), 'reuse': reuse}

            models, loss = model_function(model_factory, model_kwargs, config)
            towers.append(Tower(name_scope, device_fn(gpu), models, loss, dataset))

        # Reuse the variables for the next models
        reuse = True

    return towers

def train(session, model_factory, dataset, model_function, config, name, summary_writer):
    """ Train the classifier model """
    towers = generate_towers(session.graph, model_factory, dataset, model_function, config)

    trainer = ModelTrainer(
        session, config.batch_size, towers,
        config.decay_step, config.learning_rate,
        summary_writer, name=name)

    trainer.train(
        config.num_epochs, config.metric_frequency,
        config.validation_frequency, config.trace_frequency)

    return trainer.towers[0].models[0]

def test_model(session, classifier, model, dataset):
    """ Test the model's error rate """
    error_rate = 0
    batch_size = 256
    graph_utils.set_training(False)
    steps = int(np.ceil(len(dataset.test) / batch_size))

    for step in xrange(steps):
        offset = (step * batch_size)
        test_data = dataset.test[offset:(offset + batch_size), ...]

        feed_dict = {}
        model.feed(feed_dict, test_data)
        test_data.images = session.run(model.output_tensor, feed_dict=feed_dict)

        classifier.feed(feed_dict, test_data)
        error_rate += session.run(classifier.error_rate, feed_dict=feed_dict)

    return error_rate/steps

def main(argv=None):  # pylint: disable=unused-argument
    """Run with the specified arguments."""
    config = config_utils.load(
        OVERRIDES,
        config_file=FLAGS.config_file,
        defaults_file=FLAGS.defaults_file)

    # Setup the training directory
    if tf.gfile.Exists(FLAGS.traindir):
        tf.gfile.DeleteRecursively(FLAGS.traindir)
    tf.gfile.MakeDirs(FLAGS.traindir)

    graph = graph_utils.trainable_graph(FLAGS.num_gpus)
    summary_writer = tf.summary.FileWriter(FLAGS.traindir, graph=graph)

    if FLAGS.seed:
        graph.seed = FLAGS.seed
        np.random.seed(FLAGS.seed)

    data_factory = DatasetFactory(FLAGS.datadir)
    data_factory.register('cifar10', CIFAR10DataLoader)
    data_factory.register('mnist', MNISTDataLoader)
    data_factory.register('slice', SliceDataLoader)
    dataset = data_factory.load_data(FLAGS.dataset, FLAGS.validation_percent)

    model_factory = ModelFactory(dataset)
    model_factory.register('classification', ClassificationModelFactory)
    model_factory.register('generation', GenerationModelFactory)
    model_factory.register('vae', VariationalAutoencoderModelFactory)

    session = tf.Session(graph=graph, config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))

    with session.as_default():
        session.run(graph_utils.initialize_training(graph))

        model_config = config_utils.override(config['vae'], {'denoising': True})
        dvae = train(
            session, model_factory,
            dataset, generate_vae,
            model_config, 'dvae_trainer', summary_writer)

        model_config = config_utils.override(config['vae'], {'denoising': False})
        vae = train(
            session, model_factory,
            dataset, generate_vae,
            model_config, 'vae_trainer', summary_writer)

        model_config = config['classifier']
        classifier = train(
            session, model_factory,
            dataset, generate_classifier,
            model_config, 'class_trainer', summary_writer)

        error_rate = test_model(session, classifier, dvae, dataset)
        print('DVAE Error Rate: {:.2%}'.format(error_rate))

        error_rate = test_model(session, classifier, vae, dataset)
        print('VAE Error Rate: {:.2%}'.format(error_rate))

if __name__ == '__main__':
    FLAGS, OVERRIDES = config_utils.parse_commandline()
    tf.app.run()
