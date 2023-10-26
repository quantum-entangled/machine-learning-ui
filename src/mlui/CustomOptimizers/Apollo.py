import tensorflow as tf
import numpy as np


class Apollo(tf.keras.optimizers.legacy.Optimizer):
    """Optimizer that implements the Apollo algorithm.

    Apollo is a nonconvex stochastic optimization method
    that incorporates the curvature of the loss function by
    approximating the Hessian using a diagonal matrix.
    It is a quasi-Newton approach.
    Reference:
    - [Xuezhe Ma, 2020] <https://arxiv.org/abs/2009.13586>_

    Parameters
    ----------
    learning_rate : float, default: 1e-2
        The learning rate.
    beta : float, default: 0.9
        The coefficient used for computing running averages of gradient.
    epsilon : float, default: 1e-4
        The term added to the denominator to improve numerical stability.
    rebound : str, default: “constant”
        The recified bound for diagonal hessian: “constant” | “belief”.
    warmup : int, default: 500
        The number of warmup steps.
    init_lr : float, default: 1e-5
        The initial learning rate for warmup.
    weight_decay : float, default: 0.0
        The weight decay coefficient.
    weight_decay_type : str, default: “L2”
        The type of weight decay:  “L2” | “decoupled” | “stable”.
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            beta=0.9,
            epsilon=1e-4,
            init_lr=None,
            warmup=500,
            rebound="constant",
            weight_decay=0.0,
            weight_decay_type="L2",
            name="Apollo",
            **kwargs
    ):
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if init_lr is None:
            init_lr=0.00001

        super(Apollo, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta", beta)
        self.epsilon = epsilon
        self.init_lr=init_lr
        self.warmup=warmup
        self.rebound_type = rebound
        self._set_hyper("weight_decay", weight_decay)
        self.weight_decay_type = weight_decay_type

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg_grad")
        for var in var_list:
            self.add_slot(var, "approx_hessian")
        for var in var_list:
            self.add_slot(var, "update")


    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        curr_lr = self._decayed_lr(var_dtype)
        weight_decay = self._get_hyper("weight_decay", var_dtype)
        wdt = self.weight_decay_type
        beta = self._get_hyper("beta", var_dtype)
        eps = tf.convert_to_tensor(self.epsilon, var_dtype)
        warmup_t=tf.convert_to_tensor(self.warmup, var_dtype)
        init_lr_t=tf.convert_to_tensor(self.init_lr, var_dtype)
        step = tf.cast(self.iterations + 1, var_dtype)

        # Calculate current lr
        if step <= warmup_t:
            curr_lr = init_lr_t + (curr_lr - init_lr_t) * step / warmup_t

        # Perform step weight decay
        if weight_decay != 0 and wdt == "L2":
            grad = grad + var*weight_decay

        exp_avg_grad = self.get_slot(var, "exp_avg_grad")
        B = self.get_slot(var, "approx_hessian")
        d_p = self.get_slot(var, "update")

        bias_correction = 1 - beta ** step
        alpha = (1 - beta) / bias_correction

        # calc the diff grad
        delta_grad = grad - exp_avg_grad
        if self.rebound_type == "belief":
            rebound = tf.norm(delta_grad, ord=np.inf)
        else:
            rebound = 0.01
            eps = eps / rebound

        # Update the running average grad
        exp_avg_grad_t = exp_avg_grad.assign_add(delta_grad * alpha, use_locking=self._use_locking)

        denom = tf.norm(d_p, ord=4)+eps
        d_p_t = d_p / denom
        v_sq = d_p_t ** 2
        delta = tf.math.reduce_sum((delta_grad / denom) * d_p_t) * (-alpha) - tf.math.reduce_sum(B * v_sq)

        # Update B
        B_t = B.assign_add(v_sq * delta, use_locking=self._use_locking)

        # calc direction of parameter updates
        if self.rebound_type == 'belief':
            denom = tf.math.maximum(tf.math.abs(B_t), rebound) + eps / alpha
        else:
            denom = tf.math.maximum(tf.math.abs(B_t), rebound)

        d_p_t = exp_avg_grad_t / denom

        # Perform step weight decay (decoupled)
        if weight_decay != 0 and wdt != "L2":
            if wdt == "stable":
                weight_decay_t = weight_decay/tf.reduce_mean(denom)
            else:
                weight_decay_t = weight_decay
            d_p_t = d_p.assign(d_p_t + var*weight_decay_t)

        var.assign_add(d_p_t * -curr_lr, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise RuntimeError("Apollo does not support sparse gradients.")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta": self._serialize_hyperparameter("beta"),
                "epsilon": self.epsilon,
                "rebound": self.rebound_type,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
            }
        )
        return config