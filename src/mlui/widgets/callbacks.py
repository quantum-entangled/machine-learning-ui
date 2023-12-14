from abc import ABC, abstractmethod

import streamlit as st

from mlui.types.classes import CallbackParams, EarlyStoppingParams


class CallbackWidget(ABC):
    """Base class for a callback's widget."""

    @abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abstractmethod
    def params(self) -> CallbackParams:
        """Callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class EarlyStopping(CallbackWidget):
    """EarlyStopping callback's widget."""

    def __init__(self) -> None:
        self.min_delta = st.number_input(
            "Min delta:",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            format="%e",
        )
        self.patience = st.number_input(
            "Patience:", min_value=0, max_value=50, value=10, step=1
        )

    @property
    def params(self) -> EarlyStoppingParams:
        """EarlyStopping callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"min_delta": float(self.min_delta), "patience": int(self.patience)}


class TerminateOnNaN(CallbackWidget):
    """TerminateOnNaN callback's widget."""

    def __init__(self) -> None:
        pass

    @property
    def params(self) -> CallbackParams:
        """TerminateOnNaN callback's parameters."""
        return {}
