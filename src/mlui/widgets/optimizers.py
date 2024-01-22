import abc

import streamlit as st

import mlui.types.classes as t


class OptimizerWidget(abc.ABC):
    """Base class for a widget of the optimizer."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize the widgets of parameters."""
        ...

    @property
    @abc.abstractmethod
    def params(self) -> t.OptimizerParams:
        """Adjustable parameters of the optimizer."""


class Adam(OptimizerWidget):
    """Widget class for the Adam optimizer."""

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
        return {
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
        }


class RMSprop(OptimizerWidget):
    """Widget class for the RMSprop optimizer."""

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
        return {
            "learning_rate": float(self.learning_rate),
            "rho": float(self.rho),
            "momentum": float(self.momentum),
        }


class SGD(OptimizerWidget):
    """Widget class for the SGD optimizer."""

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
        return {
            "learning_rate": float(self.learning_rate),
            "momentum": float(self.momentum),
        }
