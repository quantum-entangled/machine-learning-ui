import abc

import streamlit as st

import mlui.types.classes as t


class CallbackWidget(abc.ABC):
    """Base class for a widget of the callback."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize the widgets of parameters."""
        ...

    @property
    @abc.abstractmethod
    def params(self) -> t.CallbackParams:
        """Adjustable parameters of the callback."""


class EarlyStopping(CallbackWidget):
    """Widget class for the EarlyStopping callback."""

    def __init__(self) -> None:
        self._min_delta = st.number_input(
            "Min delta:",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            format="%e",
        )
        self._patience = st.number_input(
            "Patience:", min_value=0, max_value=50, value=10, step=1
        )

    @property
    def params(self) -> t.EarlyStoppingParams:
        return {"min_delta": float(self._min_delta), "patience": int(self._patience)}


class TerminateOnNaN(CallbackWidget):
    """Widget class for the TerminateOnNaN callback."""

    def __init__(self) -> None:
        pass

    @property
    def params(self) -> t.CallbackParams:
        return {}
