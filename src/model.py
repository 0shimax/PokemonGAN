import tensorflow as tf
import ops


class BiRNN(tf.keras.Model):
    def __init__(self, output_size, embedding_size, n_hidden,
                 stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=None):
        self.wrod_embeddings = tf.keras.layers.Embedding(150, embedding_size)
        self.lstm_fw = tf.keras.layers.LSTM(n_hidden)
        self.lstm_bw = tf.keras.layers.LSTM(n_hidden, go_backwards=True)
        self.fw_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bw_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.lin = tf.keras.layers.Dense(output_size)

    def bi_directional(self, input_):
        # https://github.com/keras-team/keras/issues/2838
        rnn_fwd1 = self.lstm_fw(input_)
        rnn_fwd1 = self.fw_dropout(rnn_fwd1)
        rnn_bwd1 = self.self.lstm_fw(rnn_fwd1)
        rnn_bwd1 = self.bw_dropout(rnn_bwd1)
        rnn_bidir1 = tf.keras.layers.merge([rnn_fwd1, rnn_bwd1], mode='concat')
        return rnn_bidir1

    def call(self, input_):
        embedded_word_output = self.wrod_embeddings(input_)
        lstm_output = self.bi_directional(embedded_word_output)
        out = self.lin(lstm_output)
        return out


class Sampler(tf.keras.Model):
    def __init__(self, options, t_z, t_text_embedding):
        super().__init__()

        s = options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        """ dim 2400 -> 256 """

        self.bi_rnn = BiRNN(options['t_dim'], options['word_dim'],
                            options['rnn_hidden'])

        self.g_h0_lin = \
            tf.keras.layers.Dense(options['gf_dim']*8*s16*s16)
        self.g_h1 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*4, kernel_size=5, strides=2)
        self.g_h2 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*2*4, kernel_size=5, strides=2)
        self.g_h3 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*1, kernel_size=5, strides=2)
        self.g_h4 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=5, strides=2)

        self.g_bn0 = tf.keras.layers.BatchNormalization()
        self.g_bn1 = tf.keras.layers.BatchNormalization()
        self.g_bn2 = tf.keras.layers.BatchNormalization()
        self.g_bn3 = tf.keras.layers.BatchNormalization()

    def call(self, t_z, t_text_embedding):
        reduced_text_embedding = self.bi_rnn(t_text_embedding)
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        z_ = self.g_h0_lin(z_concat)
        h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim']*8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = self.g_h1(h0)
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = self.g_h2(h1)
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = self.g_h3(h2)
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = self.g_h4(h3)
        return (tf.tanh(h4)/2. + 0.5)


# GENERATOR IMPLEMENTATION based on:
#     https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Generator(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        s = options['image_size']
        s2, s4, s8, self.s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        self.bi_rnn = BiRNN(options['t_dim'], options['word_dim'],
                            options['rnn_hidden'])
        self.g_h0_lin = tf.keras.layers.Dense(options['gf_dim']*8*s16*s16)

        self.g_h1 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*4, kernel_size=5, strides=2)
        self.g_h2 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*2, kernel_size=5, strides=2)
        self.g_h3 = tf.keras.layers.Conv2DTranspose(
            filters=options['gf_dim']*1, kernel_size=5, strides=2)
        self.g_h4 = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=5, strides=2)

        self.g_bn0 = tf.keras.layers.BatchNormalization()
        self.g_bn1 = tf.keras.layers.BatchNormalization()
        self.g_bn2 = tf.keras.layers.BatchNormalization()
        self.g_bn3 = tf.keras.layers.BatchNormalization()

    def call(self, t_z, t_text_embedding):
        reduced_text_embedding = self.bi_rnn(t_text_embedding)
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)

        z_ = self.g_h0_lin(z_concat)

        h0 = tf.reshape(z_,
            [-1, self.s16, self.s16, self.options['gf_dim'] * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = self.g_h1(h0)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = self.g_h2(h1)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = self.g_h3(h2)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = self.g_h4(h3)
        return (tf.tanh(h4)/2. + 0.5)

# DISCRIMINATOR IMPLEMENTATION based on :
#    https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class Discriminator(tf.keras.Model):
    def __init__(self, options, reuse=False):
        self.d_h0_conv = tf.keras.layers.Conv2D(options['df_dim'],
            kernel_size=5, strides=2, activation=tf.nn.leaky_relu)
        self.d_h1_conv = tf.keras.layers.Conv2D(options['df_dim']*2,
            kernel_size=5, strides=2)
        self.d_h2_conv = tf.keras.layers.Conv2D(options['df_dim']*4,
            kernel_size=5, strides=2)
        self.d_h3_conv = tf.keras.layers.Conv2D(options['df_dim']*8,
            kernel_size=5, strides=2)
        self.d_h3_conv_new = tf.keras.layers.Conv2D(self.options['df_dim']*8,
            kernel_size=1, strides=1)

        self.bi_rnn = BiRNN(options['t_dim'], options['word_dim'],
                            options['rnn_hidden'])
        self.d_h3_lin = tf.keras.layers.Dense(1)

        self.d_bn1 = tf.keras.layers.BatchNormalization()
        self.d_bn2 = tf.keras.layers.BatchNormalization()
        self.d_bn3 = tf.keras.layers.BatchNormalization()
        self.d_bn4 = tf.keras.layers.BatchNormalization()

    def call(self, image, t_text_embedding):
        h0 = self.d_h0_conv(image)  # 32
        h1 = tf.nn.leaky_relu(self.d_bn1(self.d_h1_conv(h0)))  # 16
        h2 = tf.nn.leaky_relu(self.d_bn2(self.d_h2_conv(h1)))  # 8
        h3 = tf.nn.leaky_relu(self.d_bn3(self.d_h3_conv(h2)))  # 4

        # ADD TEXT EMBEDDING TO THE NETWORK
        # TODO: replace this part with charcter base RNN
        reduced_text_embeddings = self.bi_rnn(t_text_embedding)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 2, 2, 1],
                                   name='tiled_embeddings')
        h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')
        h3_new = tf.nn.leaky_relu(self.d_bn4(self.d_h3_conv_new(h3_concat)))

        h4 = self.d_h3_lin(
            tf.reshape(h3_new, [self.options['batch_size'], -1]))
        return tf.nn.sigmoid(h4), h4


class GAN(tf.keras.Model):
    """
    OPTIONS
    z_dim : Noise dimension 100
    t_dim : Text feature dimension 256
    image_size : Image Dimension 64
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    caption_vector_length : Caption Vector Length 2400
    batch_size : Batch Size 64
    """

    def __init__(self, options):
        super().__init__()
        self.options = options
        self.generator = Generator(options)
        self.discriminator = Discriminator(options)


    def call(self, t_real_image, t_wrong_image, t_real_caption, t_z):
        fake_image = self.generator(t_z, t_real_caption)

        disc_real_image, disc_real_image_logits = \
            self.discriminator(t_real_image, t_real_caption)
        disc_wrong_image, disc_wrong_image_logits = \
            self.discriminator(t_wrong_image, t_real_caption, reuse=True)
        disc_fake_image, disc_fake_image_logits = \
            self.discriminator(fake_image, t_real_caption, reuse=True)

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake_image_logits,
                labels=tf.ones_like(disc_fake_image)))

        d_loss1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_real_image_logits,
                labels=tf.ones_like(disc_real_image)))
        d_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_wrong_image_logits,
                labels=tf.zeros_like(disc_wrong_image)))
        d_loss3 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake_image_logits,
                labels=tf.zeros_like(disc_fake_image)))
        d_loss = d_loss1 + d_loss2 + d_loss3

        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]

        input_tensors = {
            't_real_image': t_real_image,
            't_wrong_image': t_wrong_image,
            't_real_caption': t_real_caption,
            't_z': t_z
        }

        # variables = {
        #     'd_vars': d_vars,
        #     'g_vars': g_vars
        # }

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        outputs = {
            'generator': fake_image
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3': d_loss3,
            'disc_real_image_logits': disc_real_image_logits,
            'disc_wrong_image_logits': disc_wrong_image,
            'disc_fake_image_logits': disc_fake_image_logits
        }

        return input_tensors, variables, loss, outputs, checks

    def build_generator(self, image_size, t_real_caption, t_z):
        fake_image = self.sampler(t_z, t_real_caption)

        input_tensors = {
            't_real_caption': t_real_caption,
            't_z': t_z
        }

        outputs = {
            'generator': fake_image
        }
        return input_tensors, outputs
