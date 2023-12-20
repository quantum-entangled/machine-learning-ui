from keras.optimizers.optimizer import Optimizer
import tensorflow as tf
from keras.optimizers import utils as optimizer_utils
from tensorflow.python.platform import tf_logging as logging
import numpy as np
from tensorflow import gradients


class AdaHessian(Optimizer):
    """Optimizer that implements the AdaHessian algorithm.

        AdaHessian is an optimization algorithm for training
        neural networks that uses a gradient and a Hessian to
        update model parameters. It effectively controls the
        learning rate and overcomes the problem of ill-conditioned
        loss functions, which improves learning results.
        Reference:
        - [Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa,
        Kurt Keutzer, Michael W. Mahoney, 2021] <https://arxiv.org/abs/2006.00719>_

        Parameters
        ----------
        learning_rate : float, default: 0.15
            The learning rate.
        beta_1 : float, default: 0.9
            The exponential decay rate for the 1st moment estimates.
        beta_2 :float, default: 0.999
            The exponential decay rate for the 2nd moment estimates.
        weight_decay : float, default: 0.0
            The weight decay.
        hessian_power : float, default: 0.5
            Hessian power to control the optimizer.
        epsilon : float, default: 1e-4
            The term added to the denominator to improve numerical stability.
        """

    def __init__(
            self,
            learning_rate=0.15,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-4,
            weight_decay=0.0,
            hessian_power=0.5,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="AdaHessian",
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.hessian_power = hessian_power

    def build(self, var_list):
        """Initialize optimizer variables.

        AdaHessian optimizer has 2 types of variables: exponential moving
        average of gradient values, exponential moving average of Hessian
        diagonal square values.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        self._exp_avgs = []
        self._exp_hessian_diag_sqs = []
        for var in var_list:
            self._exp_avgs.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="exp_avg"
                )
            )

            self._exp_hessian_diag_sqs.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="exp_hessian_diag_sq"
                )
            )

    def compute_gradients(self, loss, var_list, tape=None):
        """Compute gradients and approximation of hessian trace
         of loss on trainable variables.

         Parameters
         ----------
         loss
             The loss tensor.
         var_list
             List of model variables.
         tape : tf.GradientTape, default: None
             Gradient tape to compute gradients automatically.
        """

        if not callable(loss) and tape is None:
            raise ValueError(
                "`tape` is required when a `Tensor` loss is passed. "
                f"Received: loss={loss}, tape={tape}."
            )
        if tape is None:
            tape = tf.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()

        grads = tape.gradient(loss, var_list)
        v = [np.random.uniform(0, 1, size=p.shape) for p in var_list]
        for vi in v:
            vi[vi < 0.5] = -1
            vi[vi >= 0.5] = 1
        v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]

        vprod = tf.reduce_sum([tf.reduce_sum(vi * grad) for vi, grad in zip(v, grads)])

        Hv = gradients(vprod, var_list)

        hessians = [tf.abs(Hvi * vi) for Hvi, vi in zip(Hv, v)]
        return list(zip(grads, hessians, var_list))

    def minimize(self, loss, var_list, tape=None):
        """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`.

        Parameters
        ----------
        loss
            The loss tensor.
        var_list
            List of model variables.
        tape : tf.GradientTape, default: None
             Gradient tape to compute gradients automatically.
        """

        grads_hess_and_vars = self.compute_gradients(loss, var_list, tape)
        self.apply_gradients(grads_hess_and_vars)

    def apply_gradients(self, grads_hess_and_vars, name=None, skip_gradients_aggregation=False, **kwargs,):
        """Apply gradients to variables.

        Parameters
        ----------
        grads_hess_and_vars
            List of `(gradient, hessian, variable)` triplets.
        """

        experimental_aggregate_gradients = kwargs.pop(
            "experimental_aggregate_gradients", True
        )
        if not skip_gradients_aggregation and experimental_aggregate_gradients:
            grads_hess_and_vars = self.aggregate_gradients(grads_hess_and_vars)
        self._compute_current_learning_rate()
        grads_hess_and_vars = list(grads_hess_and_vars)
        if len(grads_hess_and_vars) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return self._iterations
        grads, hessians, trainable_variables = zip(*grads_hess_and_vars)
        scope_name = name or self.name or "optimizer"
        with tf.name_scope(scope_name):
            with tf.init_scope():
                # Lift variable creation to init scope to avoid environment
                # issues.
                self.build(trainable_variables)
        grads_hess_and_vars = list(zip(grads, hessians, trainable_variables))
        grads_hess_and_vars = self._filter_grads(grads_hess_and_vars)
        if len(list(grads_hess_and_vars)) == 0:
            # Check again after filtering gradients.
            return self._iterations

        grads, hessians, trainable_variables = zip(*grads_hess_and_vars)

        grads = self._clip_gradients(grads)
        grads = self._deduplicate_sparse_grad(grads)
        self._apply_weight_decay(trainable_variables)
        grads_hess_and_vars = list(zip(grads, hessians, trainable_variables))
        iteration = self._internal_apply_gradients(grads_hess_and_vars)

        # Apply variable constraints after applying gradients.
        for variable in trainable_variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        return iteration

    def aggregate_gradients(self, grads_hess_and_vars):
        """Aggregate gradients on all devices.

        Parameters
        ----------
        grads_hess_and_vars
            List of `(gradient, hessian, variable)` triplets.
        """

        return self._all_reduce_sum_gradients(grads_hess_and_vars)

    def _all_reduce_sum_gradients(self, grads_hess_and_vars):
        """Returns all-reduced gradients aggregated via summation.

        Parameters
        ----------
        grads_hess_and_vars
            List of `(gradient, hessian, variable)` triplets.
        """

        grads_hess_and_vars = list(grads_hess_and_vars)
        filtered_grads_hessians_and_vars = self._filter_grads(grads_hess_and_vars)
        if filtered_grads_hessians_and_vars:
            if tf.__internal__.distribute.strategy_supports_no_merge_call():
                grads = [pair[0] for pair in filtered_grads_hessians_and_vars]
                reduced = tf.distribute.get_replica_context().all_reduce(
                    tf.distribute.ReduceOp.SUM, grads
                )
            else:
                # TODO(b/183257003): Remove this branch
                reduced = tf.distribute.get_replica_context().merge_call(
                    self._all_reduce_sum_fn, args=(filtered_grads_hessians_and_vars,)
                )
        else:
            reduced = []
        # Copy 'reduced' but add None gradients back in
        reduced_with_nones = []
        reduced_pos = 0
        for g, h, v in grads_hess_and_vars:
            if g is None:
                reduced_with_nones.append((None, h, v))
            else:
                reduced_with_nones.append((reduced[reduced_pos], h, v))
                reduced_pos += 1
        assert reduced_pos == len(reduced), "Failed to add all gradients"
        return reduced_with_nones

    def _all_reduce_sum_fn(self, distribution, grads_and_vars):
        return distribution.extended.batch_reduce_to(
            tf.distribute.ReduceOp.SUM, grads_and_vars
        )

    def _filter_grads(self, grads_hess_and_vars):
        """Filter out `(grad, hess, var)` pairs that have a gradient equal to `None`.

        Parameters
        ----------
        grads_hess_and_vars
            List of `(gradient, hessian, variable)` triplets.
        """

        grads_hess_and_vars = tuple(grads_hess_and_vars)
        if not grads_hess_and_vars:
            return grads_hess_and_vars

        filtered = []
        vars_with_empty_grads = []
        for grad, hessian, var in grads_hess_and_vars:
            if grad is None:
                vars_with_empty_grads.append(var)
            else:
                filtered.append((grad, hessian, var))
        filtered = tuple(filtered)

        if not filtered:
            variable = ([v.name for _, _, v in grads_hess_and_vars],)
            raise ValueError(
                f"No gradients provided for any variable: {variable}. "
                f"Provided `grads_and_vars` is {grads_hess_and_vars}."
            )
        if vars_with_empty_grads:
            logging.warning(
                "Gradients do not exist for variables %s when minimizing the "
                "loss. If you're using `model.compile()`, did you forget to "
                "provide a `loss` argument?",
                ([v.name for v in vars_with_empty_grads]),
            )
        return filtered

    def _distributed_apply_gradients_fn(
        self, distribution, grads_hess_and_vars, **kwargs
    ):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad, hess):
            if self.jit_compile:
                return self._update_step_xla(grad, hess, var, id(self._var_key(var)))
            else:
                return self._update_step(grad, hess, var)

        for grad, hess, var in grads_hess_and_vars:
            distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad, hess,), group=False
            )

        if self.use_ema:
            _, _, var_list = zip(*grads_hess_and_vars)
            self._update_model_variables_moving_average(var_list)
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                tf.cond(
                    tf.cast(should_overwrite_model_vars, tf.bool),
                    true_fn=lambda: self._overwrite_model_variables_with_average_value(  # noqa: E501
                        var_list
                    ),
                    false_fn=lambda: None,
                )
        return self.iterations.assign_add(1)

    def _internal_apply_gradients(self, grads_hess_and_vars):

        return tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_apply_gradients_fn,
            self._distribution_strategy,
            grads_hess_and_vars,
        )

    @tf.function(jit_compile=True)
    def _update_step_xla(self, gradient, hessian, variable, key):
        """A wrapper of `update_step` to enable XLA acceleration.

        """
        return self._update_step(gradient, hessian, variable)

    def _update_step(self, gradient, hessian, variable):
        if getattr(variable, "_unique_id", None) is None:
            # Variable has no `_unique_id` if called during `model.save()`, in
            # which case we do not want to update the variable.
            return
        if self._var_key(variable) not in self._index_dict:
            raise KeyError(
                f"The optimizer cannot recognize variable {variable.name}. "
                "This usually means you are trying to call the optimizer to "
                "update different parts of the model separately. Please call "
                "`optimizer.build(variables)` with the full list of trainable "
                "variables before the training loop or use legacy optimizer "
                f"`tf.keras.optimizers.legacy.{self.__class__.__name__}."
            )
        self.update_step(gradient, hessian, variable)

    def update_step(self, gradient, hessian, variable):
        """Update step given gradient and the associated model variable."""

        lr_t = tf.cast(self.learning_rate, variable.dtype)
        weight_decay = tf.cast(self.weight_decay, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        eps = tf.cast(self.epsilon, variable.dtype)
        k = tf.cast(self.hessian_power, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)

        var_key = self._var_key(variable)
        exp_avg = self._exp_avgs[self._index_dict[var_key]]
        exp_hessian_diag_sq = self._exp_hessian_diag_sqs[self._index_dict[var_key]]

        hess_average = tf.reduce_mean(hessian)

        exp_avg.assign(exp_avg * beta_1 + gradient * (1 - beta_1))
        exp_hessian_diag_sq.assign(exp_hessian_diag_sq * beta_2 + tf.pow(hess_average, 2) * (1 - beta_2))

        bias_correction1 = 1 - beta_1 ** step
        bias_correction2 = 1 - beta_2 ** step

        denom = (
                        (tf.sqrt(exp_hessian_diag_sq) ** k) / (tf.sqrt(bias_correction2) ** k)) \
                + eps

        variable.assign_sub(lr_t * (
                exp_avg / bias_correction1 / denom
                + weight_decay * variable))

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