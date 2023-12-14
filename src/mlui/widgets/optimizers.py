from abc import ABC, abstractmethod

import streamlit as st

from mlui.types.classes import AdamParams, OptimizerParams, RMSpropParams, SGDParams


class OptimizerWidget(ABC):
    """Base class for an optimizer's widget."""

    @abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abstractmethod
    def params(self) -> OptimizerParams:
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
            min_value=0.0,
            max_value=0.5,
            value=0.001,
            step=0.001,
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

    @property
    def params(self) -> AdamParams:
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
            min_value=0.0,
            max_value=0.5,
            value=0.001,
            step=0.001,
            format="%e",
        )
        self.rho = st.number_input(
            "Discounting factor:",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.001,
            format="%.3f",
        )
        self.momentum = st.number_input(
            "Momentum:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.001,
            format="%.3f",
        )

    @property
    def params(self) -> RMSpropParams:
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
            min_value=0.0,
            max_value=0.5,
            value=0.001,
            step=0.001,
            format="%e",
        )
        self.momentum = st.number_input(
            "Momentum:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.001,
            format="%.3f",
        )

    @property
    def params(self) -> SGDParams:
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
