import tensorflow as tf
from tensorflow.python.keras import backend_config
import numpy as np


class AdaHessian(tf.keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.15,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-4,
            weight_decay=0,
            hessian_power=0.5,
            name="AdaHessian",
            **kwargs
    ):
        super(AdaHessian, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self._set_hyper("weight_decay", weight_decay)
        self.hessian_power = hessian_power

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg")
        for var in var_list:
            self.add_slot(var, "exp_hessian_diag_sq")

    def get_trace(self, params, grads):

        v = [
            2
            * tf.random.uniform(
                p.shape, maxval=1, dtype=p.dtype
            )
            - 1
            for p in params
        ]

        hvs = tf.gradients(grads, params, grad_ys=v)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.shape
            if len(param_size) <= 2:
                tmp_output = tf.abs(hv)

            elif len(param_size) == 4:
                tmp_output = tf.reduce_mean(tf.abs(hv), axis=[2, 3], keepdims=True)
            hutchinson_trace.append(tmp_output)

        return hutchinson_trace

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        weight_decay = self._get_hyper("weight_decay", var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        eps = tf.convert_to_tensor(self.epsilon, var_dtype)
        k = tf.convert_to_tensor(self.hessian_power, var_dtype)
        step = tf.cast(self.iterations + 1, var_dtype)

        exp_avg = self.get_slot(var, "exp_avg")
        exp_hessian_diag_sq = self.get_slot(var, "exp_hessian_diag_sq")

        params = [var]
        grads = [grad]

        hut_trace = self.get_trace(params, grads)

        exp_avg.assign(exp_avg * beta_1 + grad * (1 - beta_1))
        exp_hessian_diag_sq.assign(exp_hessian_diag_sq * beta_2 + tf.pow(hut_trace, 2) * (1 - beta_2))

        bias_correction1 = 1 - beta_1 ** step
        bias_correction2 = 1 - beta_2 ** step

        denom = (
                (tf.sqrt(exp_hessian_diag_sq)**k)
                / np.sqrt(bias_correction2)**k
        ) + eps

        var.assign_sub(lr_t * (
            exp_avg / bias_correction1 /denom
            + weight_decay * var
        ), use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise RuntimeError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(AdaHessian, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
            }
        )
        return config
