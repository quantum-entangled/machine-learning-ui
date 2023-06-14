import data_classes.model as model_cls
import streamlit as st

import enums.callbacks as callbacks
import managers.errors as err
import managers.model_manager as mm


def set_callbacks_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model's callbacks.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Callbacks")

    callback = str(st.selectbox("Select callback class:", list(callbacks.classes)))
    callback_cls = callbacks.classes[callback]
    callback_widget = callbacks.widgets[callback]()
    callback_params = callback_widget.params
    set_callback_btn = st.button("Set Callback")

    if set_callback_btn:
        try:
            mm.set_callback(callback_cls, callback_params, model)
            st.success("Callback is set!", icon="✅")
        except (err.NoModelError, err.SameCallbackError) as error:
            st.error(error, icon="❌")
