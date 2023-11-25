import functools

import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls


# TODO: Add docstrings and type hints
def set_classes(func):
    @functools.wraps(func)
    def inner():
        if "data" not in st.session_state:
            st.session_state.data = data_cls.Data()

        if "model" not in st.session_state:
            st.session_state.model = model_cls.Model()

        func()

    return inner


def set_task(func):
    @functools.wraps(func)
    def inner():
        if "task" not in st.session_state:
            st.session_state.task = "Training"

        func()

    return inner
