import tensorflow.compat.v2 as tf
from keras import backend
from keras.optimizers import Optimizer


class AdaBound(Optimizer):
    """
    Adabound optimizer implementation
    # References
         'Adaptive Gradient Methods with Dynamic Bound of Learning Rate
          <https://openreview.net/forum?id=Bkg3g2R9FX>'_

    Parameters
    ----------
        learning_rate : float, default: 1e-3
                        Adam learning rate.

        base_lr : float, default: None
                  Used for loading the optimizer.

        final_lr : float, default: 0.1
                   final (SGD) learning rate.

        beta1 : float, default: 0.9
                coefficient used for computingrunning averages of gradient and its square.

        beta2 : float, default: 0.999
                coefficient used for computingrunning averages of gradient and its square.

        gamma : float, default: 1e-3
                convergence speed of the bound functions.

        eps : float, default: 1e-8
              term added to the denominator to improve numerical stability.

        AMSGrad : boolean, default: False
                  whether to use the AMSGrad variant of this algorithm.
    """

    def __init__(
        self,
        learning_rate=0.001,
        base_lr=None,
        final_lr=0.1,
        beta_1=0.9,
        beta_2=0.999,
        gamma=1e-3,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdaBound",
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

        if base_lr is None:
            if isinstance(
                learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
            ):
                base_lr = learning_rate(0)
            else:
                base_lr = learning_rate
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.base_lr = base_lr
        self.fianl_lr = final_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """AdaBound optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when AMSGrad is applied).

        parameters
        ----------
            var_list
                     list of model variables to build AdaBound variables on.
        """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_variable=var, variable_name="m")
            )

            self._velocities.append(
                self.add_variable_from_reference(model_variable=var, variable_name="v")
            )

        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr_t = tf.cast(self.learning_rate, variable.dtype)
        base_lr = tf.cast(self.base_lr, variable.dtype)
        final_lr = tf.cast(self.fianl_lr, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)

        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        gamma = tf.cast(self.gamma, variable.dtype)
        epsilon_t = tf.cast(self.epsilon, variable.dtype)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(gradient.values * (1 - self.beta_1), gradient.indices)
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )

            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                denom = tf.sqrt(v_hat) + epsilon_t
            else:
                denom = tf.sqrt(v) + epsilon_t

            final_lr = final_lr * lr_t / base_lr
            lower_bound = final_lr * (1.0 - 1.0 / (gamma * local_step + 1.0))
            upper_bound = final_lr * (1.0 + 1.0 / (gamma * local_step))
            lr_t = lr_t * (tf.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power))
            lr_t = tf.clip_by_value(lr_t / denom, lower_bound, upper_bound)

            variable.assign_sub(lr_t * m)
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))

            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                denom = tf.sqrt(v_hat) + epsilon_t
            else:
                denom = tf.sqrt(v) + epsilon_t

            final_lr = final_lr * lr_t / base_lr
            lower_bound = final_lr * (1.0 - 1.0 / (gamma * local_step + 1.0))
            upper_bound = final_lr * (1.0 + 1.0 / (gamma * local_step))
            lr_t = lr_t * (tf.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power))
            lr_t = tf.clip_by_value(lr_t / denom, lower_bound, upper_bound)

            variable.assign_sub(lr_t * m)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "base_lr": self._serialize_hyperparameter("base_lr"),
                "final_lr": self._serialize_hyperparameter("final_lr"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
