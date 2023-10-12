import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.errors as err
import mlui.managers.model_manager as mm
from mlui.enums import callbacks


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

    with st.expander("Callback's Parameters"):
        callback_widget = callbacks.widgets[callback]()
        callback_params = callback_widget.params

    set_callback_btn = st.button("Set Callback")

    st.markdown("To reset all existing callbacks, please press this button:")

    reset_callbacks_btn = st.button("Reset All Callbacks")

    if set_callback_btn:
        try:
            mm.set_callback(callback_cls, callback_params, model)
            st.success("Callback is set!", icon="✅")
        except (err.NoModelError, err.SameCallbackError) as error:
            st.error(error, icon="❌")

    if reset_callbacks_btn:
        try:
            mm.reset_callbacks(model)
            st.success("Callbacks are reset!", icon="✅")
        except err.NoModelError as error:
            st.error(error, icon="❌")


def fit_model_ui(data: data_cls.Data, model: model_cls.Model) -> None:
    """Generate UI for fitting the model.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    st.header("Fit Model")

    batch_size = int(
        st.number_input("Batch size:", min_value=1, max_value=1024, value=32, step=1)
    )
    num_epochs = int(
        st.number_input(
            "Number of epochs:", min_value=1, max_value=1000, value=30, step=1
        )
    )
    val_split = float(
        st.number_input(
            "Validation split:", min_value=0.01, max_value=1.0, value=0.15, step=0.01
        )
    )
    fit_model_btn = st.button("Fit Model")
    batch_container = st.empty()
    epoch_container = st.expander("Epochs Logs")

    if fit_model_btn:
        try:
            mm.fit_model(
                batch_size,
                num_epochs,
                val_split,
                batch_container,
                epoch_container,
                data,
                model,
            )
            st.success("Training is completed!", icon="✅")
        except (
            err.NoModelError,
            err.DataNotSplitError,
            err.ModelNotCompiledError,
        ) as error:
            st.error(error, icon="❌")


def show_history_plot_ui(model: model_cls.Model) -> None:
    """Generate UI for plotting the training history.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("History Plot")

    logs = model.training_history.drop("epoch", axis=1).columns
    history_type = st.multiselect("Select logs to plot:", logs)
    schemes = ["set1", "set2", "set3"]
    color_scheme = str(st.selectbox("Select color scheme:", schemes))
    plot_history_btn = st.button("Plot History")

    if plot_history_btn:
        try:
            chart = mm.show_history_plot(history_type, color_scheme, model)
            st.altair_chart(chart, use_container_width=True)
        except (err.NoModelError, err.ModelNotTrainedError) as error:
            st.error(error, icon="❌")
