from keras.optimizers.optimizer import Optimizer
import tensorflow as tf


class MADGRAD(Optimizer):
    """Optimizer that implements the MADGRAD algorithm.

    MADGRAD is an optimization algorithm used in deep learning.
    It combines adaptive learning rates and momentum to efficiently
    navigate the loss landscape during training.
    With its ability to dynamically adjust the learning rate and smooth
    out oscillations using momentum,
    MADGRAD offers fast convergence and improved optimization performance.
    Reference:
    - [Aaron Defazio, Samy Jelassi, 2021] <https://arxiv.org/abs/2101.11075>_

    Parameters
    ----------
    learning_rate : float, default: 1e-2
        The learning rate.
    momentum : float, default: 0.0
        The momentum value in the range [0,1).
    weight_decay : float, default: 0.0
        The weight decay (L2 penalty).
    epsilon : float, default: 1e-6
        The term added to the denominator to improve numerical stability.
    """

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.0,
            epsilon=1e-6,
            weight_decay=0.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="MADGRAD",
            **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        if weight_decay != 0:
            self.use_decay = True
        else:
            self.use_decay = False
        if momentum != 0:
            self.use_momentum = True
        else:
            self.use_momentum = False

    def build(self, var_list):
        """Initialize optimizer variables.

        MADGRAD optimizer has 2 types of variables: exponential average
        of gradient values, exponential average of squared gradient.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        self._grad_sum = []
        self._grad_sum_sq = []
        for var in var_list:
            self._grad_sum.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="s"
                )
            )

            self._grad_sum_sq.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr_t = tf.cast(self.learning_rate, variable.dtype)
        momentum = tf.cast(self.momentum, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)
        decay = tf.cast(self.weight_decay, variable.dtype)

        var_key = self._var_key(variable)
        s = self._grad_sum[self._index_dict[var_key]]
        v = self._grad_sum_sq[self._index_dict[var_key]]

        ck = 1 - momentum
        lamb = lr_t * tf.sqrt(step)

        if isinstance(gradient, tf.IndexedSlices):
            #Sparse gradients.

            if self.use_momentum:
                raise RuntimeError(
                    "momentum != 0 is not compatible with "
                    "sparse gradients"
                )
            if self.use_decay:
                raise RuntimeError(
                    "weight_decay option is not "
                    "compatible with sparse gradients"
                )
            s.assign_add(s)
            s.scatter_add(
                tf.IndexedSlices(
                    gradient.values * lamb, gradient.indices
                )
            )

            v.assign_add(v)
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * lamb, gradient.indices
                )
            )

            rms = tf.pow(v, 1 / 3) + self.epsilon
            x0 = variable + tf.divide(s, rms)
            var = x0 - tf.divide(s, rms)

            variable.assign(var)

        else:
            #Dence gradients.

            if self.use_decay:
                gradient += decay * variable

            if not self.use_momentum:
                rms = tf.pow(v, 1/3) + self.epsilon
                x0 = variable + tf.divide(s, rms)
            else:
                x0 = variable

            s.assign_add(lamb * gradient)
            v.assign_add(lamb * (gradient * gradient))

            rms = tf.pow(v, 1/3) + self.epsilon

            if not self.use_momentum:
                var = x0 - tf.divide(s, rms)
            else:
                z = x0 - tf.divide(s, rms)
                var = variable * (1 - ck) + z * ck

            variable.assign(var)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'learning_rate': self._serialize_hyperparameter(
                    self._learning_rate
                ),
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'epsilon': self.epsilon,
            }
        )
        return config
