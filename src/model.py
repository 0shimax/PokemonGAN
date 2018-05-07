import tensorflow as tf


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

        s = options['image_size']
        self.s2, self.s4, self.s8, self.s16 = \
            int(s/2), int(s/4), int(s/8), int(s/16)

        # self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
        #                     options['rnn_hidden'], options['voc_dim'],
        #                     options['embedding_matrix'])

        self.rnn = RNN(options['rnn_output_dim'], options['embedded_size'],
                       options['rnn_hidden'], options['voc_dim'],
                       options['embedding_matrix'])

        self.g_fc1 = tf.keras.layers.Dense(1024)
        # if useing depth_to_space, channel size is 3*4*4
        self.g_fc2 = tf.keras.layers.Dense(options['gf_dim']*2*self.s4*self.s4)

        # TODO: rewrite to tf.keras.layers.UpSampling2D
        # self.g_h1 = tf.keras.layers.Conv2DTranspose(
        #     filters=options['gf_dim']*4, kernel_size=5,
        #     strides=2, padding="same")
        # self.g_h2 = tf.keras.layers.Conv2DTranspose(
        #     filters=options['gf_dim']*2, kernel_size=5,
        #     strides=2, padding="same")
        self.g_h3 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim'], kernel_size=5,
            strides=2, padding="same")
        self.g_h4 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=5,
            strides=2, padding="same",
            activation=tf.nn.tanh)
        self.g_conv1 = tf.keras.layers.Conv2D(options['gf_dim'],
            kernel_size=3, strides=1, padding="same")
        self.g_conv2 = tf.keras.layers.Conv2D(3,
            kernel_size=3, strides=1, padding="same",
            activation=tf.nn.tanh)

        self.g_norm0 = tf.keras.layers.BatchNormalization()
        self.g_norm1 = tf.keras.layers.BatchNormalization()
        self.g_norm2 = tf.keras.layers.BatchNormalization()
        self.g_norm3 = tf.keras.layers.BatchNormalization()

        # self.g_norm0 = InstanceNormalization()
        # self.g_norm1 = InstanceNormalization()
        # self.g_norm2 = InstanceNormalization()
        # self.g_norm3 = InstanceNormalization()

    def __call__(self, t_z, t_text_embedding):
        # reduced_text_embedding = self.rnn(t_text_embedding)
        # z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        # z_ = self.g_h0_lin(z_concat)

        h_z = tf.nn.elu(self.g_norm0(self.g_fc1(t_z)))
        h_z = self.g_fc2(h_z)
        h0 = tf.reshape(h_z,
            [-1, self.s4, self.s4, self.options['gf_dim'] * 2])

        h0 = tf.nn.elu(self.g_norm1(h0))

        # h1 = self.g_h1(h0)
        # h1 = tf.nn.elu(self.g_norm1(h1))
        #
        # h2 = self.g_h2(h1)
        # h2 = tf.nn.elu(self.g_norm2(h2))

        h3 = tf.nn.elu(self.g_norm3(self.g_h3(h0)))
        return self.g_h4(h3)

        # h3 = tf.depth_to_space(h0, 2)
        # h3 = self.g_conv1(h3)
        # h3 = tf.nn.elu(self.g_norm3(h3))
        # return self.g_conv2(tf.depth_to_space(h3, 2))


# DISCRIMINATOR IMPLEMENTATION based on :
#    https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Discriminator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        self.d_h0_conv = tf.keras.layers.Conv2D(options['df_dim'],
            kernel_size=5, strides=2, padding="same")
        self.d_h1_conv = tf.keras.layers.Conv2D(options['df_dim']*2,
            kernel_size=5, strides=2, padding="same")
        self.d_h2_conv = tf.keras.layers.Conv2D(options['df_dim']*4,
            kernel_size=5, strides=2, padding="same")
        self.d_h3_conv = tf.keras.layers.Conv2D(options['df_dim']*8,
            kernel_size=5, strides=2, padding="same")
        # self.d_h3_conv_new = tf.keras.layers.Conv2D(options['df_dim']*8,
        #     kernel_size=1, strides=1, padding="same")

        # self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
        #                     options['rnn_hidden'], options['voc_dim'],
        #                     options['embedding_matrix'])

        self.rnn = RNN(options['rnn_output_dim'], options['embedded_size'],
                       options['rnn_hidden'], options['voc_dim'],
                       options['embedding_matrix'])

        self.flatten = tf.keras.layers.Flatten()
        self.d_fc1 = tf.keras.layers.Dense(1024)
        self.d_fc2 = tf.keras.layers.Dense(1)

        self.d_norm1 = tf.keras.layers.BatchNormalization()
        self.d_norm2 = tf.keras.layers.BatchNormalization()
        # self.d_norm3 = tf.keras.layers.BatchNormalization()
        # self.d_norm4 = tf.keras.layers.BatchNormalization()

        # self.d_norm1 = InstanceNormalization()
        # self.d_norm2 = InstanceNormalization()
        self.d_norm3 = InstanceNormalization()
        self.d_norm4 = InstanceNormalization()

    def _global_average_pooling(self, x):
        for _ in range(2):
            x = tf.reduce_mean(x, axis=1)
        return x

    def _add_noise(self, h, sigma=0.2, training=False):
        if training:
            return h + sigma * tf.random_normal(h.shape)
        else:
            return h

    def __call__(self, image, t_text_embedding, training=True):
        n_batch, _, _, _ = image.shape

        h = self._add_noise(image)
        h0 = tf.nn.elu(self._add_noise(self.d_h0_conv(h),
                                       training=training))  # 32
        h1 = tf.nn.elu(
            self._add_noise(self.d_norm1(self.d_h1_conv(h0)),
                            training=training))  # 16
        h2 = tf.nn.elu(
            self._add_noise(self.d_norm2(self.d_h2_conv(h1)),
                            training=training))  # 8
        h3 = tf.nn.elu(
            self._add_noise(self.d_norm3(self.d_h3_conv(h2)),
                            training=training))  # 4

        # h3_pooled = self._global_average_pooling(h3)
        h3_flatten = self.flatten(h3)

        # reduced_text_embeddings = self.rnn(t_text_embedding)
        # h3_concat = tf.concat([h3_flatten, reduced_text_embeddings], 1)
        # hf1 = tf.nn.leaky_relu(self.d_norm4(self.d_fc1(h3_concat)))

        # hf1 = tf.nn.elu(
        #     self._add_noise(self.d_norm4(self.d_fc1(h3_flatten))))
        hf1 = tf.nn.elu(
            self._add_noise(self.d_fc1(h3_flatten), training=training))
        hf2 = self.d_fc2(hf1)

        return hf2
