import abc
from typing import TypedDict

import streamlit as st
import tensorflow as tf


class OptimizerParams(TypedDict):
    """Base type annotation for an optimizer's parameters."""


class AdamParams(OptimizerParams):
    """Type annotation for the Adam optimizer."""

    learning_rate: float
    beta_1: float
    beta_2: float


class RMSpropParams(OptimizerParams):
    """Type annotation for the RMSprop optimizer."""

    learning_rate: float
    rho: float
    momentum: float


class SGDParams(OptimizerParams):
    """Type annotation for the SGD optimizer."""

    learning_rate: float
    momentum: float


class AdaModParams(OptimizerParams):
    """Type annotation for the AdaMod optimizer."""

    learning_rate: float
    beta_1: float
    beta_2: float
    beta_3: float


class ApolloParams(OptimizerParams):
    """Type annotation for the Apollo optimizer."""

    learning_rate: float
    beta: float
    weight_decay: float
    weight_decay_type: str


class LAMBParams(OptimizerParams):
    """Type annotation for the LAMB optimizer."""

    learning_rate: float
    beta_1: float
    beta_2: float


class LookaheadParams(OptimizerParams):
    """Type annotation for the Lookahead optimizer."""

    optimizer: tf.keras.optimizers.Optimizer


class RAdamParams(OptimizerParams):
    """Type annotation for the RAdam optimizer."""

    learning_rate: float
    beta_1: float
    beta_2: float


class MADGRADParams(OptimizerParams):
    """Type annotation for the MADGRAD optimizer."""

    learning_rate: float
    momentum: float
    weight_decay: float

class LARSParams(OptimizerParams):
    """Type annotation for the LARS optimizer."""

    learning_rate: float
    momentum: float
    weight_decay: float
    dampening: float
    nesterov: bool


class OptimizerWidget(abc.ABC):
    """Base class for an optimizer's widget."""

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def params(self) -> OptimizerParams:
        """Optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class Adam(OptimizerWidget):
    """Adam optimizer widget."""

    def __init__(self) -> None:
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.5,
                value=0.001,
                step=0.001,
                format="%e",
            )
        )
        self.beta_1 = float(
            st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        )
        self.beta_2 = float(
            st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.001,
                format="%.3f",
            )
        )

    @property
    def params(self) -> AdamParams:
        """Optimizer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
        }


class RMSprop(OptimizerWidget):
    """RMSprop optimizer widget."""

    def __init__(self):
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.5,
                value=0.001,
                step=0.001,
                format="%e",
            )
        )
        self.rho = float(
            st.number_input(
                "Discounting factor:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        )
        self.momentum = float(
            st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        )

    @property
    def params(self) -> RMSpropParams:
        return {
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "momentum": self.momentum,
        }


class SGD(OptimizerWidget):
    """SGD optimizer widget."""

    def __init__(self):
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.5,
                value=0.001,
                step=0.001,
                format="%e",
            )
        )
        self.momentum = float(
            st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        )

    @property
    def params(self) -> SGDParams:
        return {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
        }


class AdaMod(OptimizerWidget):
    """AdaMod optimizer widget."""

    def __init__(self):
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        )
        self.beta_1 = float(
            st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        )
        self.beta_2 = float(
            st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.001,
                format="%.3f",
            )
        )
        self.beta_3 = float(
            st.number_input(
                "Decay 3:",
                min_value=0.0,
                max_value=1.0,
                value=0.995,
                step=0.001,
                format="%.3f",
            )
        )

    @property
    def params(self) -> AdaModParams:
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
        }


class Apollo(OptimizerWidget):
    """Apollo optimizer widget."""

    def __init__(self):
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        )
        self.beta = float(
            st.number_input(
                "Beta:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%e",
            )
        )
        self.weight_decay = float(
            st.number_input(
                "Weight Decay:",
                min_value=0.0,
                max_value=0.001,
                value=0.0,
                step=0.00005,
                format="%e",
            )
        )
        self.weight_decay_type = str(
            st.selectbox(
                "Weight Decay Type:",
                ("L2", "Decoupled", "Stable"),
            )
        )

    @property
    def params(self) -> ApolloParams:
        return {
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "weight_decay": self.weight_decay,
            "weight_decay_type": self.weight_decay_type,
        }


class LAMB(OptimizerWidget):
    """LAMB optimizer widget."""

    def __init__(self):
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        )
        self.beta_1 = float(
            st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%.3f",
            )
        )
        self.beta_2 = float(
            st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.005,
                format="%.3f",
            )
        )

    @property
    def params(self) -> LAMBParams:
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
        }


class Lookahead(OptimizerWidget):
    """Lookahead optimizer widget."""

    def __init__(self):
        self.optimizer = tf.keras.optimizers.Optimizer(
            st.selectbox(
                "Optimizer:",
                options=(tf.keras.optimizers.SGD(), tf.keras.optimizers.Adam(), tf.keras.optimizers.RMSprop(),
                 tf.keras.optimizers.Adamax(), tf.keras.optimizers.Adagrad())
            )
        )

    @property
    def params(self) -> LookaheadParams:
        return {
            "optimizer": self.optimizer
        }


class RAdam(OptimizerWidget):
    """RAdam optimizer widget."""

    def __init__(self) -> None:
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.005,
                format="%e",
            )
        )
        self.beta_1 = float(
            st.number_input(
                "Decay 1:",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.005,
                format="%.3f",
            )
        )
        self.beta_2 = float(
            st.number_input(
                "Decay 2:",
                min_value=0.0,
                max_value=1.0,
                value=0.999,
                step=0.005,
                format="%.3f",
            )
        )

    @property
    def params(self) -> RAdamParams:
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
        }


class MADGRAD(OptimizerWidget):
    """MADGRAD optimizer widget."""

    def __init__(self) -> None:
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        )
        self.momentum = float(
            st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=0.999,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        )
        self.weight_decay = float(
            st.number_input(
                "Weight decay:",
                min_value=0.0,
                max_value=0.999,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        )

    @property
    def params(self) -> MADGRADParams:
        return {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }


class LARS(OptimizerWidget):
    """LARS optimizer widget."""

    def __init__(self) -> None:
        self.learning_rate = float(
            st.number_input(
                "Learning rate:",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%e",
            )
        )
        self.momentum = float(
            st.number_input(
                "Momentum:",
                min_value=0.0,
                max_value=0.999,
                value=0.9,
                step=0.001,
                format="%.3f",
            )
        )
        self.weight_decay = float(
            st.number_input(
                "Weight decay:",
                min_value=0.0,
                max_value=0.999,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        )
        self.dampening = float(
            st.number_input(
                "Dampening for momentum:",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f",
            )
        )
        self.nesterov = bool(
            st.selectbox(
                "Nesterov momentum:",
                (False, True),
            )
        )

    @property
    def params(self) -> LARSParams:
        return {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "dampening": self.dampening,
            "nesterov": self.nesterov,
        }

