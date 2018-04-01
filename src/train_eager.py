import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
import argparse
import random
import os
import time

import model
from load_data import read_data_sets


def main(args):
    tfe.enable_eager_execution()
    args.no_gpu = True

    (device, data_format) = ('/gpu:0', 'channels_first')
    if args.no_gpu or tfe.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    data = read_data_sets('./data', 'pokemon_images', 'pokemon.csv')
    dataset = (
        tf.data.Dataset.from_tensor_slices(data.train.images).shuffle(60000)
        .batch(args.batch_size))

    model_options = {
        'rnn_hidden': args.rnn_hidden,
        'word_dim': args.word_dim,
        'z_dim': args.z_dim,
        't_dim': args.t_dim,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length
    }

    # Create the models and optimizers.
    model_objects = {
        'generator': model.Generator(model_options),
        'discriminator': model.Discriminator(model_options),
        'generator_optimizer': tf.train.AdamOptimizer(args.learning_rate),
        'discriminator_optimizer': tf.train.AdamOptimizer(args.learning_rate),
        'step_counter': tf.train.get_or_create_global_step()}

    train(args, model_objects, device, dataset)


def train(args, model_objects, device, dataset):
    # Prepare summary writer and checkpoint info
    summary_writer = tf.contrib.summary.create_summary_file_writer(
        args.out_dir, flush_millis=1000)
    checkpoint_prefix = os.path.join(args.checkpoint_dir, 'ckpt')
    latest_cpkt = tf.train.latest_checkpoint(args.checkpoint_dir)
    if latest_cpkt:
        print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint = tfe.Checkpoint(**model_objects)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)

    with tf.device(device):
        for i_epoch in range(args.epochs):
            print("epoch" + str(i_epoch))

            start = time.time()
            with summary_writer.as_default():
                train_one_epoch(dataset=dataset,
                                log_interval=args.log_interval,
                                noise_dim=args.noise, **model_objects)
            end = time.time()
            checkpoint.save(checkpoint_prefix)
            print('\nTrain time for epoch #%d (step %d): %f' %
                  (checkpoint.save_counter.numpy(),
                   checkpoint.step_counter.numpy(),
                   end - start))


def train_one_epoch(generator, discriminator, generator_optimizer,
                    discriminator_optimizer, dataset, step_counter,
                    log_interval, noise_dim):
    """Trains `generator` and `discriminator` models on `dataset`.
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        generator_optimizer: Optimizer to use for generator.
        discriminator_optimizer: Optimizer to use for discriminator.
        dataset: Dataset of images to train on.
        step_counter: An integer variable, used to write summaries regularly.
        log_interval: How many steps to wait between logging and collecting
          summaries.
        noise_dim: Dimension of noise vector to use.
    """

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    for (batch_index, images) in enumerate(tfe.Iterator(dataset)):
        with tf.device('/cpu:0'):
            tf.assign_add(step_counter, 1)

        with tf.contrib.summary.record_summaries_every_n_global_steps(
            args.log_interval, global_step=step_counter):
            current_batch_size = images.shape[0]
            noise = tf.random_uniform(
                shape=[current_batch_size, noise_dim],
                minval=-1.,
                maxval=1.,
                seed=batch_index)

            with tfe.GradientTape(persistent=True) as g:
                generated_images = generator(noise)
                tf.contrib.summary.image(
                    'generated_images',
                    tf.reshape(generated_images, [-1, 28, 28, 1]),
                    max_images=10)

                discriminator_gen_outputs = discriminator(generated_images)
                discriminator_real_outputs = discriminator(images)
                discriminator_loss_val = \
                    discriminator_loss(discriminator_real_outputs,
                                       discriminator_gen_outputs)
                total_discriminator_loss += discriminator_loss_val

                generator_loss_val = generator_loss(discriminator_gen_outputs)
                total_generator_loss += generator_loss_val

            generator_grad = g.gradient(generator_loss_val, generator.variables)
            discriminator_grad = g.gradient(discriminator_loss_val,
                                          discriminator.variables)

            generator_optimizer.apply_gradients(
                zip(generator_grad, generator.variables))
            discriminator_optimizer.apply_gradients(
                zip(discriminator_grad, discriminator.variables))

            if log_interval and batch_index > 0 \
                    and batch_index % log_interval == 0:
                print('Batch #%d\tAverage Generator Loss: %.6f\t'
                      'Average Discriminator Loss: %.6f' %
                      (batch_index, total_generator_loss/batch_index,
                       total_discriminator_loss/batch_index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_hidden', type=int, default=200,
                        help='Number of nodes in the rnn hidden layer')

    parser.add_argument('--z_dim', type=int, default=100,
                        help='Noise dimension')

    parser.add_argument('--word_dim', type=int, default=256,
                        help='Word embedding matrix dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                        help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch Size')

    parser.add_argument('--image_size', type=int, default=32,
                        help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis \
                              for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=20,
                        help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Data Directory')

    parser.add_argument('--out_dir', type=str, default="./results",
                        help='Output Directory')

    parser.add_argument('--checkpoint_dir', type=str, default="./results/ckpt",
                        help='Checkpoint Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                        help='Max number of epochs')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Log interval')
    parser.add_argument('--noise', type=int, default=30,
                        help='Noise dimention')

    parser.add_argument('--save_every', type=int, default=30,
                        help='Save Model/Samples every x iterations \
                              over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers",
                        help='Dat set: MS-COCO, flowers')
    args = parser.parse_args()

    main(args)
