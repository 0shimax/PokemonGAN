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
                 stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=None):
        super().__init__()

        self.word_embeddings = tf.keras.layers.Embedding(
            voc_dim, embedded_size)
        self.bidirectional = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(n_hidden))
        self.lin = tf.keras.layers.Dense(output_size)

    def call(self, input_):
        embedded_word_output = self.word_embeddings(input_)
        lstm_output = self.bidirectional(embedded_word_output)
        out = self.lin(lstm_output)
        return out


# GENERATOR IMPLEMENTATION based on:
#     https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Generator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        s = options['image_size']
        self.s2, self.s4, self.s8, self.s16 = \
            int(s/2), int(s/4), int(s/8), int(s/16)

        self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
                            options['rnn_hidden'], options['voc_dim'])
        self.g_h0_lin = tf.keras.layers.Dense(
            options['gf_dim']*8*self.s16*self.s16, activation=tf.tanh)

        self.g_h1 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*4, kernel_size=5,
            strides=2, padding="same")
        self.g_h2 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*2, kernel_size=5,
            strides=2, padding="same")
        self.g_h3 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim'], kernel_size=5,
            strides=2, padding="same")
        self.g_h4 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=5,
            strides=2, padding="same",
            activation=tf.nn.sigmoid)

        # self.g_bn0 = tf.keras.layers.BatchNormalization()
        # self.g_bn1 = tf.keras.layers.BatchNormalization()
        # self.g_bn2 = tf.keras.layers.BatchNormalization()
        # self.g_bn3 = tf.keras.layers.BatchNormalization()

        self.g_norm0 = InstanceNormalization()
        self.g_norm1 = InstanceNormalization()
        self.g_norm2 = InstanceNormalization()
        self.g_norm3 = InstanceNormalization()

    def __call__(self, t_z, t_text_embedding):
        reduced_text_embedding = self.bi_rnn(t_text_embedding)
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        z_ = self.g_h0_lin(z_concat)

        h0 = tf.reshape(z_,
            [-1, self.s16, self.s16, self.options['gf_dim'] * 8])

        h0 = tf.nn.elu(self.g_norm0(h0))

        h1 = self.g_h1(h0)
        h1 = tf.nn.elu(self.g_norm1(h1))

        h2 = self.g_h2(h1)
        h2 = tf.nn.elu(self.g_norm2(h2))

        h3 = self.g_h3(h2)
        h3 = tf.nn.elu(self.g_norm3(h3))
        return self.g_h4(h3)


# DISCRIMINATOR IMPLEMENTATION based on :
#    https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Discriminator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        self.d_h0_conv = tf.keras.layers.Conv2D(options['df_dim'],
            kernel_size=5, strides=2, activation=tf.nn.leaky_relu,
            padding="same")
        self.d_h1_conv = tf.keras.layers.Conv2D(options['df_dim']*2,
            kernel_size=5, strides=2, padding="same")
        self.d_h2_conv = tf.keras.layers.Conv2D(options['df_dim']*4,
            kernel_size=5, strides=2, padding="same")
        self.d_h3_conv = tf.keras.layers.Conv2D(options['df_dim']*8,
            kernel_size=5, strides=2, padding="same")
        # self.d_h3_conv_new = tf.keras.layers.Conv2D(options['df_dim']*8,
        #     kernel_size=1, strides=1, padding="same")

        self.bi_rnn = BiRNN(options['rnn_output_dim'], options['embedded_size'],
                            options['rnn_hidden'], options['voc_dim'])

        self.flatten = tf.keras.layers.Flatten()
        self.d_fc1 = tf.keras.layers.Dense(1024)
        self.d_fc2 = tf.keras.layers.Dense(1)

        # self.d_bn1 = tf.keras.layers.BatchNormalization()
        # self.d_bn2 = tf.keras.layers.BatchNormalization()
        # self.d_bn3 = tf.keras.layers.BatchNormalization()
        # self.d_bn4 = tf.keras.layers.BatchNormalization()
        # self.d_bn5 = tf.keras.layers.BatchNormalization()

        self.d_norm1 = InstanceNormalization()
        self.d_norm2 = InstanceNormalization()
        self.d_norm3 = InstanceNormalization()
        self.d_norm4 = InstanceNormalization()

    def _global_average_pooling(self, x):
        for _ in range(2):
            x = tf.reduce_mean(x, axis=1)
        return x

    def _add_noise(self, h, sigma=0.2):
        return h + sigma * tf.random_normal(h.shape)

    def __call__(self, image, t_text_embedding, training=True):
        n_batch, _, _, _ = image.shape
        if training:
            h = self._add_noise(image)
        else:
            h = image

        h0 = self.d_h0_conv(h)  # 32
        h1 = tf.nn.elu(self.d_norm1(self.d_h1_conv(h0)))  # 16
        h2 = tf.nn.elu(self.d_norm2(self.d_h2_conv(h1)))  # 8
        h3 = tf.nn.elu(self.d_norm3(self.d_h3_conv(h2)))  # 4

        # h3_pooled = self._global_average_pooling(h3)
        h3_flatten = self.flatten(h3)
        reduced_text_embeddings = self.bi_rnn(t_text_embedding)

        h3_concat = tf.concat([h3_flatten, reduced_text_embeddings], 1)
        hf1 = tf.nn.elu(self.d_norm4(self.d_fc1(h3_concat)))
        hf2 = self.d_fc2(hf1)

        return hf2
