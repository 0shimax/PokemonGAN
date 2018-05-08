import tensorflow as tf
import numpy as np


class InstanceNormalization(tf.keras.Model):
    """
    ref: https://qiita.com/t-ae/items/39daefcdbe8bf927e4f3
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        input_shape = inputs.shape

        if len(input_shape) == 2:
            mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
            return (inputs - mean) / tf.sqrt(var + tf.keras.backend.epsilon())
        elif len(input_shape) == 4:
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            return (inputs - mean) / tf.sqrt(var + tf.keras.backend.epsilon())
        else:
            raise ValueError("Not valid")


class BiRNN(tf.keras.Model):
    def __init__(self, output_size, embedded_size, n_hidden, voc_dim,
                 embedding_matrix,
                 stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=None):
        super().__init__()

        self.word_embeddings = tf.keras.layers.Embedding(
            voc_dim, embedded_size,
            weights=[embedding_matrix], trainable=True)
        # self.word_embeddings.set_weights([embedding_matrix])
        self.bidirectional = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(n_hidden))
        self.lin = tf.keras.layers.Dense(output_size)

    def call(self, input_):
        embedded_word_output = self.word_embeddings(input_)
        lstm_output = self.bidirectional(embedded_word_output)
        out = self.lin(lstm_output)
        return out


class RNN(tf.keras.Model):
    def __init__(self, output_size, embedded_size, n_hidden, voc_dim,
                 embedding_matrix,
                 stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=None):
        super().__init__()

        self.word_embeddings = tf.keras.layers.Embedding(
            voc_dim, embedded_size,
            weights=[embedding_matrix], trainable=True)

        self.gru = tf.keras.layers.GRU(n_hidden)
        self.fc = tf.keras.layers.Dense(output_size, activation=tf.nn.elu)

    def call(self, input_):
        embedded_word_output = self.word_embeddings(input_)
        lstm_output = self.gru(embedded_word_output)
        return self.fc(lstm_output)


# GENERATOR IMPLEMENTATION based on:
#     https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Generator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        self.c4, self.c8, self.c16, self.c32, self.c64 = 512, 256, 128, 64, 32
        self.s4 = 4
        # s = options['image_size']
        # self.s2, self.s4, self.s8, self.s16 = \
        #     int(s/2), int(s/4), int(s/8), int(s/16)

        # self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
        #                     options['rnn_hidden'], options['voc_dim'],
        #                     options['embedding_matrix'])

        self.rnn = RNN(options['rnn_output_dim'], options['embedded_size'],
                       options['rnn_hidden'], options['voc_dim'],
                       options['embedding_matrix'])

        self.g_fc1 = tf.keras.layers.Dense(
            self.s4 * self.s4 * self.c4,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.g_h1 = tf.keras.layers.Conv2DTranspose(
            filters=self.c8, kernel_size=5,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            strides=2, padding="same")
        self.g_h2 = tf.keras.layers.Conv2DTranspose(
            filters=self.c16, kernel_size=5,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            strides=2, padding="same")
        self.g_h3 = tf.keras.layers.Conv2DTranspose(
            filters=self.c32, kernel_size=5,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            strides=2, padding="same")
        self.g_h4 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=5,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            strides=2, padding="same")

    def __call__(self, t_z, t_text_embedding, is_train=True):
        # reduced_text_embedding = self.rnn(t_text_embedding)
        # z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        # z_ = self.g_h0_lin(z_concat)

        h_z = self.g_fc1(t_z)
        h0 = tf.reshape(h_z, [-1, self.s4, self.s4, self.c4])
        h0 = tf.contrib.layers.batch_norm(h0, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h0 = tf.nn.relu(h0)

        h1 = self.g_h1(h0)
        h1 = tf.contrib.layers.batch_norm(h1, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h1 = tf.nn.relu(h1)

        h2 = self.g_h2(h1)
        h2 = tf.contrib.layers.batch_norm(h2, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h2 = tf.nn.relu(h2)

        h3 = self.g_h3(h2)
        h3 = tf.contrib.layers.batch_norm(h3, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h3 = tf.nn.relu(h3)

        h4 = self.g_h4(h3)
        h4 = tf.contrib.layers.batch_norm(h4, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h4 = tf.nn.relu(h4)
        return h4


class Discriminator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.c2, self.c4, self.c8, self.c16 = 64, 128, 256, 512

        self.d_h0_conv = tf.keras.layers.Conv2D(self.c2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            kernel_size=5, strides=2, padding="same")
        self.d_h1_conv = tf.keras.layers.Conv2D(self.c4,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            kernel_size=5, strides=2, padding="same")
        self.d_h2_conv = tf.keras.layers.Conv2D(self.c8,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            kernel_size=5, strides=2, padding="same")
        self.d_h3_conv = tf.keras.layers.Conv2D(self.c16,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            kernel_size=5, strides=2, padding="same")

        # self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
        #                     options['rnn_hidden'], options['voc_dim'],
        #                     options['embedding_matrix'])

        self.rnn = RNN(options['rnn_output_dim'], options['embedded_size'],
                       options['rnn_hidden'], options['voc_dim'],
                       options['embedding_matrix'])

        self.flatten = tf.keras.layers.Flatten()
        self.d_fc = tf.keras.layers.Dense(1)

    def __call__(self, image, t_text_embedding, is_train=True):
        n_batch, _, _, _ = image.shape

        h1 = self.d_h0_conv(image)  # 32
        h1 = tf.contrib.layers.batch_norm(h1, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h1 = tf.nn.leaky_relu(h1)

        h2 = self.d_h1_conv(h1)  # 16
        h2 = tf.contrib.layers.batch_norm(h2, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h2 = tf.nn.leaky_relu(h2)

        h3 = self.d_h2_conv(h2)  # 8
        h3 = tf.contrib.layers.batch_norm(h3, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h3 = tf.nn.leaky_relu(h3)

        h4 = self.d_h3_conv(h3)  # 4
        h4 = tf.contrib.layers.batch_norm(h4, is_training=is_train,
                                          epsilon=1e-5, decay=0.9,
                                          updates_collections=None)
        h4 = tf.nn.leaky_relu(h4)

        h4_flatten = self.flatten(h4)

        # reduced_text_embeddings = self.rnn(t_text_embedding)
        # h3_concat = tf.concat([h3_flatten, reduced_text_embeddings], 1)

        return self.d_fc(h4_flatten)
