import abc

import streamlit as st

import mlui.types.classes as t


class CallbackWidget(abc.ABC):
    """Base class for a callback's widget."""

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def params(self) -> t.CallbackParams:
        """Callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class EarlyStopping(CallbackWidget):
    """EarlyStopping callback's widget."""

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
        """EarlyStopping callback's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"min_delta": float(self._min_delta), "patience": int(self._patience)}


class TerminateOnNaN(CallbackWidget):
    """TerminateOnNaN callback's widget."""

    def __init__(self) -> None:
        pass

    @property
    def params(self) -> t.CallbackParams:
        """TerminateOnNaN callback's parameters."""
        return {}
