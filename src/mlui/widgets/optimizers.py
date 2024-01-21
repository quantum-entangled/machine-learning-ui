import abc

import streamlit as st
import tensorflow as tf

import mlui.types.classes as t


class OptimizerWidget(abc.ABC):
    """Base class for an optimizer's widget."""

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def params(self) -> t.OptimizerParams:
        """Optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class Adam(OptimizerWidget):
    """Adam optimizer's widget."""

    def __init__(self) -> None:
        self.learning_rate = st.number_input(
            "Learning rate:",
            min_value=1e-6,
            max_value=1.0,
            value=1e-3,
            step=1e-3,
            format="%e",
        )
        self.beta_1 = st.number_input(
            "Decay 1:",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=1e-3,
            format="%.3f",
        )
        self.beta_2 = st.number_input(
            "Decay 2:",
            min_value=0.0,
            max_value=1.0,
            value=0.999,
            step=1e-3,
            format="%.3f",
        )

    @property
    def params(self) -> t.AdamParams:
        """Adam optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
        }


class RMSprop(OptimizerWidget):
    """RMSprop optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
            "Learning rate:",
            min_value=1e-6,
            max_value=1.0,
            value=1e-3,
            step=1e-3,
            format="%e",
        )
        self.rho = st.number_input(
            "Discounting factor:",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=1e-3,
            format="%.3f",
        )
        self.momentum = st.number_input(
            "Momentum:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=1e-3,
            format="%.3f",
        )

    @property
    def params(self) -> t.RMSpropParams:
        """RMSprop optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "rho": float(self.rho),
            "momentum": float(self.momentum),
        }


class SGD(OptimizerWidget):
    """SGD optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
            "Learning rate:",
            min_value=1e-6,
            max_value=1.0,
            value=1e-3,
            step=1e-3,
            format="%e",
        )
        self.momentum = st.number_input(
            "Momentum:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=1e-3,
            format="%.3f",
        )

    @property
    def params(self) -> t.SGDParams:
        """SGD optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "momentum": float(self.momentum),
        }


class AdaMod(OptimizerWidget):
    """AdaMod optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        self.beta_1 = st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        self.beta_2 = st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.001,
                format="%.3f",
            )
        self.beta_3 = st.number_input(
                "Decay 3:",
                min_value=0.0,
                max_value=1.0,
                value=0.995,
                step=0.001,
                format="%.3f",
            )

    @property
    def params(self) -> t.AdaModParams:
        """AdaMod optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
            "beta_3": float(self.beta_3),
        }


class Apollo(OptimizerWidget):
    """Apollo optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        self.beta = st.number_input(
                "Beta:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%.3f",
            )
        self.weight_decay = st.number_input(
                "Weight Decay:",
                min_value=0.0,
                max_value=0.001,
                value=0.0,
                step=0.00005,
                format="%.3f",
            )
        self.weight_decay_type = st.selectbox(
                "Weight Decay Type:",
                ("L2", "Decoupled", "Stable"),
            )

    @property
    def params(self) -> t.ApolloParams:
        """Apollo optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta": float(self.beta),
            "weight_decay": float(self.weight_decay),
            "weight_decay_type": str(self.weight_decay_type),
        }


class LAMB(OptimizerWidget):
    """LAMB optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        self.beta_1 = st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%.3f",
            )
        self.beta_2 = st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.005,
                format="%.3f",
            )

    @property
    def params(self) -> t.LAMBParams:
        """LAMB optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
        }


class Lookahead(OptimizerWidget):
    """Lookahead optimizer's widget."""

    def __init__(self):
        self.optimizer = st.selectbox(
                "Optimizer:",
                options=(tf.keras.optimizers.SGD(), tf.keras.optimizers.Adam(), tf.keras.optimizers.RMSprop(),
                 tf.keras.optimizers.Adamax(), tf.keras.optimizers.Adagrad())
            )

    @property
    def params(self) -> t.LookaheadParams:
        """Lookahead optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "optimizer": tf.keras.optimizers.Optimizer(self.optimizer)
        }


class RAdam(OptimizerWidget):
    """RAdam optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        self.beta_1 = st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%.3f",
            )
        self.beta_2 = st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.005,
                format="%.3f",
            )

    @property
    def params(self) -> t.RAdamParams:
        """RAdam optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
        }


class MADGRAD(OptimizerWidget):
    """MADGRAD optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        self.momentum = st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=0.999,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        self.weight_decay = st.number_input(
                "Weight decay:",
                min_value=0.0,
                max_value=0.999,
                value=0.0,
                step=0.001,
                format="%.3f",
            )

    @property
    def params(self) -> t.MADGRADParams:
        """MADGRAD optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "momentum": float(self.momentum),
            "weight_decay": float(self.weight_decay),
        }


class LARS(OptimizerWidget):
    """LARS optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        self.momentum = st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=0.999,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        self.weight_decay = st.number_input(
                "Weight decay:",
                min_value=0.0,
                max_value=0.999,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        self.dampening = st.number_input(
                "Dampening for momentum:",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        self.nesterov = st.selectbox(
                "Nesterov momentum:",
                (False, True),
            )

    @property
    def params(self) -> t.LARSParams:
        """LARS optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "momentum": float(self.momentum),
            "weight_decay": float(self.weight_decay),
            "dampening": float(self.dampening),
            "nesterov": bool(self.nesterov),
        }


class AdaHessian(OptimizerWidget):
    """AdaHessian optimizer's widget."""

    def __init__(self):
        self.learning_rate = st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.5,
                value=0.15,
                step=0.005,
                format="%e",
            )
        self.beta_1 = st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        self.beta_2 = st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.001,
                format="%.3f",
            )
        self.weight_decay = st.number_input(
                "Weight decay:",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f",
            )

    @property
    def params(self) -> t.AdaHessianParams:
        """AdaHessian optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
            "weight_decay": float(self.weight_decay),
        }
