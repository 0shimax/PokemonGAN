import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
from model import discriminator, generator
from load_data import process_data
random.seed(555)
np.random.seed(555)


HEIGHT, WIDTH, CHANNEL = 64, 64, 3
BATCH_SIZE = 64
EPOCH = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '15'
version = 'newPokemon'
newPoke_path = './' + version


def train():
    random_dim = 100
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train)

    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]


    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')
    dLoss = None
    gLoss = None
    for i in range(EPOCH):
        print(i)
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                #wgan clip weights
                sess.run(d_clip)

                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)

        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))
        if i%50 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')

            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)


# TODO: wrongも後で入れる
def discriminator_loss(discriminator_real_outputs,
                       # discriminator_wrong_outputs,
                       discriminator_gen_outputs):

    batchsize, _ = discriminator_real_outputs.shape
    batchsize = tf.cast(batchsize, tf.float32)
    loss_on_real =\
        tf.reduce_sum(tf.nn.softplus(-discriminator_real_outputs))
    loss_on_generated =\
        tf.reduce_sum(tf.nn.softplus(discriminator_gen_outputs))
    # loss_on_wrong =\
    #     tf.reduce_sum(tf.nn.softplus(discriminator_wrong_outputs))

    # d_loss = loss_on_real + (loss_on_generated + loss_on_wrong) / 2.
    d_loss = loss_on_real + loss_on_generated

    tf.contrib.summary.scalar('discriminator_loss', d_loss)
    return d_loss


def generator_loss(discriminator_fake_outputs):
    batchsize, _ = discriminator_fake_outputs.shape
    batchsize = tf.cast(batchsize, tf.float32)
    g_loss =\
        tf.reduce_sum(tf.nn.softplus(-discriminator_fake_outputs))

    tf.contrib.summary.scalar('generator_loss', g_loss)
    return g_loss


if __name__ == '__main__':
    train()
