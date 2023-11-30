from keras.optimizers.optimizer import Optimizer
import tensorflow as tf
import numpy as np


class LARS(Optimizer):
    """Optimizer that implements the LARS algorithm.

    Layer-wise Adaptive Rate Scaling, is an optimization algorithm widely used in deep learning.
    It is designed to improve the convergence speed and performance of neural network models during the training process.
    LARS combines the benefits of adaptive learning rates and layer-wise parameter scaling.
    Reference:
    - [Yang You, Igor Gitman, Boris Ginsburg, 2017] <https://arxiv.org/abs/1708.03888>_

    Parameters
    ----------
    learning_rate : float, default: 1e-3
       The learning rate.
    momentum : float, default: 0.9
       The momentum factor.
    dampening : float, default: 0.0
       The dampening for momentum.
    epsilon : float, default: 1e-8
       The term added to the denominator to improve numerical stability.
    weight_decay : float, default: 0.0
       The weight decay (L2 penalty).
    nesterov : bool, default: False
       The enables Nesterov momentum.
    trust_coefficient : float, default: 0.001
       The trust coefficient for computing LR.
    """

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0,
            dampening=0,
            nesterov=False,
            trust_coefficient=0.001,
            epsilon=1e-8,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="LARS",
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
        self.dampening = dampening
        self.nesterov = nesterov
        self.trust_coefficient = trust_coefficient
        self.epsilon = epsilon
        if momentum != 0:
            self.use_momentum = True
        else:
            self.use_momentum = False
        if weight_decay !=0:
            self.use_weight_decay = True
        else:
            self.use_weight_decay = False

    def build(self, var_list):
        """Initialize optimizer variables.

        LARS optimizer has one variable `momentums`.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr_t = tf.cast(self.learning_rate, variable.dtype)
        dampening = tf.cast(self.dampening, variable.dtype)
        weight_decay = tf.cast(self.weight_decay, variable.dtype)
        momentum = tf.cast(self.momentum, variable.dtype)

        m = None
        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]

        d_p = gradient
        w_norm = tf.norm(variable, ord=2)
        grad_norm = tf.norm(gradient, ord=2)

        # lars scaling + weight decay part
        if self.use_weight_decay:
            d_p += variable * weight_decay

        lars_lr = lr_t * tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(grad_norm, 0),
                     (self.trust_coefficient * w_norm / (grad_norm + w_norm * weight_decay + self.epsilon)), 1.0),
            1.0)
        d_p += variable * weight_decay
        d_p *= lars_lr

        if self.use_momentum:
            m_t = tf.multiply(m, momentum)
            m_t = tf.add(m_t, tf.multiply(gradient, 1 - dampening))

            if self.nesterov:
                d_p += momentum * m_t
            else:
                d_p = m_t

            m.assign(m_t)

        variable.assign_sub(lr_t * d_p)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "dampening": self.dampening,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "nesterov": self.nesterov,
                "trust_coefficient": self.trust_coefficient,
                "epsilon": self.epsilon,
            }
        )
        return config
