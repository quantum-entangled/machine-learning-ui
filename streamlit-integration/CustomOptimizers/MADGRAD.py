import tensorflow as tf
from tensorflow.python.keras import backend_config
import numpy as np


class MADGRAD(tf.keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        epsilon=1e-06,
        name="MADGRAD",
        **kwargs
    ):
        super(MADGRAD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "s")
        for var in var_list:
            self.add_slot(var, "grad_sum_sq")
        #for var in var_list:
            #self.add_slot(var, "x0")

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum = self._get_hyper("momentum", var_dtype)
        step = tf.cast(self.iterations + 1, var_dtype)
        decay = self._get_hyper("weight_decay", var_dtype)
        eps = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr_t += eps

        vk = self.get_slot(var, "grad_sum_sq")
        sk = self.get_slot(var, "s")
        #x_0 = self.get_slot(var, "x0")

        ck = 1 - momentum
        lamb = lr_t*tf.math.sqrt(step)

        # Apply weight decay
        if decay != 0:
            grad += decay*var

        if momentum == 0:
            # Compute x_0 from other known quantities
            rms = tf.pow(vk, 1/3) + eps
            x0 = var + tf.divide(sk, rms)
        else:
            x0 = var


        # Accumulate first and second moments
        sk_plus_1 = sk.assign_add(lamb * grad, use_locking=self._use_locking)
        vk_plus_1 = vk.assign_add(lamb * (grad * grad), use_locking=self._use_locking)

        rms = tf.pow(vk_plus_1, 1/3) + eps

        if momentum == 0:
            var_t = x0 - tf.divide(sk_plus_1, rms)
        else:
            z = x0 - tf.divide(sk_plus_1, rms)

            var_t = var*(1 - ck) + z*ck

        return var.assign(var_t, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise RuntimeError('This implementation does not support sparse gradients.')

    def get_config(self):
        config = super(MADGRAD, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
        })
        return config