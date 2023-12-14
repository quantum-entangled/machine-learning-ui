import functools

import streamlit as st

from mlui.classes.data import Data
from mlui.classes.model import CreatedModel, UploadedModel


# TODO: Add docstrings and type hints
def set_classes(func):
    @functools.wraps(func)
    def inner():
        data = st.session_state.get("data")
        model = st.session_state.get("model")
        task = st.session_state.get("task")

        if not data:
            st.session_state.data = Data()

        if not model and task == "Train":
            st.session_state.model = CreatedModel()
        elif not model:
            st.session_state.model = UploadedModel()

        func()

    return inner


def set_task(func):
    @functools.wraps(func)
    def inner():
        task = st.session_state.get("task")

        if not task:
            st.session_state.task = "Train"

        func()

    return inner
