import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.legacy import interfaces


class WNAdam(tf.keras.optimizers):
    """WNAdam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    # References
        - WNGrad paper - https://arxiv.org/abs/1803.02865
    """

    def __init__(self, lr=0.1, beta_1=0.9, **kwargs):
        super(WNAdam, self).__init__(**kwargs)
        with tf.keras.backend.name_scope(self.__class__.__name__):
            self.iterations = tf.keras.backend.variable(0, dtype='int64', name='iterations')
            self.lr = tf.keras.backend.variable(lr, name='lr')
            self.beta_1 = tf.keras.backend.variable(beta_1, name='beta_1')

    @tf.keras.legacy.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [tf.keras.backend.update_add(self.iterations, 1)]

        lr = self.lr

        t = tf.keras.backend.cast(self.iterations, tf.keras.backend.floatx()) + 1
        # Algorithm 4 initializations:
        # momentum accumulator is initialized with 0s
        ms = [tf.keras.backend.zeros(tf.keras.backend.int_shape(p), dtype=tf.keras.backend.dtype(p)) for p in params]
        # b parameter is initialized with 1s
        bs = [tf.keras.backend.ones(tf.keras.backend.int_shape(p), dtype=tf.keras.backend.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + bs

        for p, g, m, b in zip(params, grads, ms, bs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            b_t = b + tf.keras.backend.square(lr) * tf.keras.backend.square(g) / b
            # note: paper has the K.pow as t - 1, but this nans out when t = 1
            p_t = p - (lr / b_t) * m_t / (1 - tf.keras.backend.pow(self.beta_1, t))

            self.updates.append(tf.keras.backend.update(m, m_t))
            self.updates.append(tf.keras.backend.update(b, b_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(tf.keras.backend.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(tf.keras.backend.get_value(self.lr)),
                  'beta_1': float(tf.keras.backend.get_value(self.beta_1))}
        base_config = super(WNAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
