import abc
from typing import TypedDict

import streamlit as st


class OptimizerParams(TypedDict):
    """Base type annotation for an optimizer's parameters."""


class AdamParams(OptimizerParams):
    """Type annotation for the Adam optimizer."""

    learning_rate: float
    beta_1: float
    beta_2: float

class AdaboundParams(OptimizerParams):
    """Type annotation for the Adam optimizer"""
    learing_rate: float
    final_lr: float
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


class Adaboud(OptimizerWidget):
    """Adabound Optimizer widget"""
    def __init__(self) -> None:
        self.learing_rate= float(
            st.number_input(
                "learning rate",
                min_value= 0.0,
                max_value = 0.1,
                step = 0.001,
                format ="%e",
            )
        )
        self.final_lr= float(
            st.number_input(
                "learning rate",
                min_value= 0.0,
                max_value = 1.0,
                step = 0.005,
                format ="%e",
            )
        )
        self.bata_1= float(
            st.number_input(
                "learning rate",
                min_value= 0.0,
                max_value =1,
                step = 0.005,
                format ="%.3f",
            )
        )
        self.bata_2= float(
            st.number_input(
                "learning rate",
                min_value= 0.0,
                max_value =1,
                step = 0.005,
                format ="%.3f",
            )
        )

    @property
    def params(self) -> AdaboundParams:
        return{
            "learing_rate":self.learning_rate,
            "final_lr":self.final_lr,
            "beta_1":self.bata_1,
            "beta_2":self.bata_2,
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
