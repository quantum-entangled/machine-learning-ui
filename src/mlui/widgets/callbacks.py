import abc
from typing import TypedDict

import streamlit as st


class CallbackParams(TypedDict):
    """Base type annotation for a callback's parameters."""


class EarlyStoppingParams(CallbackParams):
    """Type annotation for the EarlyStopping callback."""

    min_delta: float
    patience: int


class CallbackWidget(abc.ABC):
    """Base class for a callback's widget."""

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def params(self) -> CallbackParams:
        """Callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class EarlyStopping(CallbackWidget):
    """EarlyStopping callback widget."""

    def __init__(self) -> None:
        self.min_delta = float(
            st.number_input(
                "Min delta:",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%e",
            )
        )
        self.patience = int(
            st.number_input("Patience:", min_value=0, max_value=50, value=10, step=1)
        )

    @property
    def params(self) -> EarlyStoppingParams:
        """Callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"min_delta": self.min_delta, "patience": self.patience}


class TerminateOnNaN(CallbackWidget):
    """TerminateOnNaN callback widget."""

    def __init__(self) -> None:
        pass

    @property
    def params(self) -> CallbackParams:
        return {}
