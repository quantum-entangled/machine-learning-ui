import numpy as np
import tensorflow as tf
from tensorflow.python.training.optimizer import Optimizer


class Apollo(Optimizer):
    def __init__(
            self,
            learning_rate=0.01,
            beta=0.9,
            epsilon=1e-4,
            rebound="constant",
            warmup=100,
            init_lr=None,
            weight_decay=0.0,
            weight_decay_type=None,
            name="Apollo",
            **kwargs
    ):
        if init_lr is None:
            init_lr=learning_rate/1000
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))

        super(Apollo, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta', beta)
        self._set_hyper('eps', epsilon)
        self._set_hyper('rebound', rebound)
        self._set_hyper('warmup', warmup)
        self._set_hyper('init_lr', init_lr)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('weight_decay_type', weight_decay_type)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'exp_avg_grad')
        for var in var_list:
            self.add_slot(var, 'approx_hessian')
        for var in var_list:
            self.add_slot(var, 'update')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_t = self._get_hyper('beta', var_dtype)
        eps_t = self._get_hyper('eps', var_dtype)
        rebound_t = self._get_hyper('rebound', var_dtype)
        warmup_t = self._get_hyper('warmup', var_dtype)
        init_lr_t = self._get_hyper('init_lr', var_dtype)
        weight_decay_t = self._get_hyper('weight_decay', var_dtype)
        weight_decay_type_t = self._get_hyper('weight_decay_type', var_dtype)

        exp_avg_grad = self.get_slot(var, 'exp_avg_grad')
        B = self.get_slot(var, 'approx_hessian')
        d_p = self.get_slot(var, 'update')

        # State initialization
        if tf.equal(self.iterations, 0):
            exp_avg_grad.assign(tf.zeros_like(var, dtype=var_dtype))
            B.assign(tf.zeros_like(var, dtype=var_dtype))
            d_p.assign(tf.zeros_like(var, dtype=var_dtype))

        step = tf.cast(self.iterations + 1, var_dtype)

        if step <= warmup_t:
            lr_t = init_lr_t + (lr_t - init_lr_t) * step / warmup_t

        # Update biased first moment estimate
        exp_avg_grad_t = beta_t * exp_avg_grad + (1 - beta_t) * grad

        # Update biased second raw moment estimate
        grad_diff = grad - exp_avg_grad_t
        approx_hessian_t = beta_t * approx_hessian + (1 - beta_t) * tf.square(grad_diff)

        # Compute corrected second raw moment estimate
        if rebound_t == 'constant':
            approx_hessian_t = tf.maximum(approx_hessian_t, eps_t)
        elif rebound_t == 'belief':
            approx_hessian_t = tf.abs(approx_hessian_t) + eps_t

        # Compute update direction
        update_t = exp_avg_grad_t / (tf.sqrt(approx_hessian_t) + eps_t)

        # Weight decay
        if weight_decay_t > 0.0:
            if weight_decay_type_t == 'L2':
                update_t += weight_decay_t * var
            elif weight_decay_type_t == 'decoupled':
                var_norm = tf.norm(var, ord=2)
                update_t += weight_decay_t * var / (var_norm + eps_t)
            elif weight_decay_type_t == 'stable':
                update_t += weight_decay_t * exp_avg_grad_t / (tf.sqrt(approx_hessian_t) * tf.norm(var, ord=2) + eps_t)

        # Apply update
        var_t = var - lr_t * update_t

        # Update state
        exp_avg_grad.assign(exp_avg_grad_t)
        approx_hessian.assign(approx_hessian_t)
        update.assign(update_t)
        tf.keras.optimizers.schedules.LearningRateSchedule