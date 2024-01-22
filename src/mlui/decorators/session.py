import functools

import streamlit as st

import mlui.classes.data as data
import mlui.classes.model as model
import mlui.types.classes as t


def set_state(func: t.FuncType) -> t.FuncType:
    """
    Decorator to set the initial state for a Streamlit app.

    Parameters
    ----------
    func : Callable
        Function to wrap and execute.

    Returns
    -------
    Callable
        Wrapped function.
    """

    @functools.wraps(func)
    def wrapper() -> None:
        if not st.session_state.get("task"):
            st.session_state.task = "Train"

        if not st.session_state.get("data"):
            st.session_state.data = data.Data()

        if not st.session_state.get("model"):
            st.session_state.model = model.CreatedModel()

        if not st.session_state.get("model_type"):
            st.session_state.model_type = "Created"

        func()

    return wrapper
