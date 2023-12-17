import functools

import streamlit as st

import mlui.classes.data as data
import mlui.classes.model as model


# TODO: Add type hints
def set_state(func):
    @functools.wraps(func)
    def inner():
        if not st.session_state.get("task"):
            st.session_state.task = "Train"

        if not st.session_state.get("data"):
            st.session_state.data = data.Data()

        if not st.session_state.get("model"):
            st.session_state.model = model.CreatedModel()

        if not st.session_state.get("model_type"):
            st.session_state.model_type = "Created"

        func()

    return inner
