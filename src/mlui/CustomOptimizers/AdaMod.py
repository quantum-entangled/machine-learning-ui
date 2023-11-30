from keras.optimizers.optimizer import Optimizer
import tensorflow as tf


class AdaMod(Optimizer):
    """Optimizer that implements the AdaMod algorithm.

    An optimizer that adjusts the learning rates for each parameter based on historical statistics,
    preventing them from becoming excessively large and avoiding non-convergence issues,
    ultimately leading to improved performance.
    Reference:
    - [Jianbang Ding, Xuancheng Ren, Ruixuan Luo, Xu Sun, 2019] <https://arxiv.org/abs/1910.12249>_

    Parameters
    ----------
    learning_rate : float, default: 1e-3
        The learning rate.
    beta_1 : float, default: 0.9
        The exponential decay rate for the 1st moment estimates.
    beta_2 : float, default: 0.999
        The exponential decay rate for the 2nd moment estimates.
    beta_3 : float, default: 0.9999
        The smoothing coefficient for adaptive learning rates.
    epsilon : float, default: 1e-8
        The term added to the denominator to improve numerical stability.
    """

    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            beta_3=0.9995,
            epsilon=1e-7,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="AdaMod",
            **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        AdaMod optimizer has 3 types of variables: exponential moving average
        of gradient values, exponential moving average of squared gradient
        values and exponential moving average of actual learning rates.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        self._exp_avg_lr = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
            self._exp_avg_lr.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="s"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        s = self._exp_avg_lr[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )

            denom = tf.sqrt(v) + self.epsilon
            step_size = tf.fill(denom.shape, alpha)
            step_size = tf.divide(step_size, denom)
            s.assign_add(-s * (1 - self.beta_3))
            s.scatter_add(
                tf.IndexedSlices(
                    step_size.values * (1 - self.beta_3), step_size.indices
                )
            )
            step_size = tf.minimum(step_size, s)

            variable.assign_sub(m * step_size)
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))

            denom = tf.sqrt(v) + self.epsilon
            step_size = tf.fill(denom.shape, alpha)
            step_size = tf.divide(step_size, denom)
            s.assign_add((step_size - s) * (1 - self.beta_3))
            step_size = tf.minimum(step_size, s)

            variable.assign_sub(m * step_size)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_3": self.beta_3,
                "epsilon": self.epsilon,
            }
        )
        return config
