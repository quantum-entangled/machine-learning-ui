import tensorflow as tf
from tensorflow.python.keras import backend_config
import numpy as np


class LARS(tf.keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0,
            dampening=0,
            nesterov=False,
            trust_coefficient=0.001,
            epsilon=1e-8,
            name="LARS",
            **kwargs
    ):
        super(LARS, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("momentum", momentum)
        self.dampening = dampening
        self.trust_coefficient = trust_coefficient
        self.nesterov = nesterov
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "momentum_buffer")

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        eps = tf.convert_to_tensor(self.epsilon, var_dtype)
        dampening = tf.convert_to_tensor(self.dampening, var_dtype)
        weight_decay = self._get_hyper("weight_decay", var_dtype)
        momentum = self._get_hyper("momentum", var_dtype)

        m = self.get_slot(var, "momentum_buffer")
        d_p = grad
        w_norm = tf.norm(var, ord=2)
        grad_norm = tf.norm(grad, ord=2)

        # lars scaling + weight decay part
        if weight_decay !=0:
            if w_norm !=0 and grad_norm !=0:
                lars_lr = w_norm/(grad_norm + w_norm*weight_decay + eps)
                lars_lr *= self.trust_coefficient

                d_p += var*weight_decay
                d_p *= lars_lr
        if momentum != 0 :
            m_t = tf.multiply(m, momentum)
            m_t = tf.add(m_t, tf.multiply(grad, 1 - dampening))

            if self.nesterov:
                d_p += momentum*m_t
            else:
                d_p = m_t

            m.assign(m_t)

        var.assign_sub(lr_t * d_p, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise RuntimeError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(LARS, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "dampening": self.dampening,
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "nesterov": self.nesterov,
            "trust_coefficient": self.trust_coefficient,
            "epsilon": self.epsilon,
        })
        return config
