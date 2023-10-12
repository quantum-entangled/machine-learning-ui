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

    if not model.trained:
        st.info(
            "You will be able to view the training history plot "
            "once you train the model.",
            icon="💡",
        )
        return

    opts = model.training_history.columns.drop("epoch")
    df_len = len(model.training_history)
    Y = st.multiselect("Select Y column(s):", opts)
    params = dict()

    with st.expander("Chart's Parameters"):
        schemes = ("set1", "set2", "set3")
        legend_orients = (
            "left",
            "right",
            "top",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        )
        legend_directions = ("vertical", "horizontal")
        params["scheme"] = str(st.selectbox("Select color scheme:", schemes))
        params["X_ticks"] = st.number_input(
            "Number of X-axis ticks:", 0, df_len + 100, df_len, 1
        )
        params["Y_ticks"] = st.number_input(
            "Number of Y-axis ticks:", 0, df_len + 100, df_len, 1
        )
        params["X_domain"] = st.slider(
            "X-axis domain range:",
            -100.0,
            df_len + 100.0,
            (0.5, df_len + 0.5),
            1.0,
        )
        params["X_title"] = st.text_input(
            "X-axis title:", "Epoch", 30, placeholder="No title if None"
        )
        params["Y_title"] = st.text_input(
            "Y-axis title:", "Value", 30, placeholder="No title if None"
        )
        params["legend_orient"] = st.select_slider(
            "Legend orientation:", legend_orients, "bottom"
        )
        params["legend_direction"] = st.select_slider(
            "Legend direction:", legend_directions, "horizontal"
        )
        params["legend_title"] = st.text_input(
            "Legend title:", None, 30, placeholder="No title if None"
        )
        params["height"] = st.number_input("Plot height:", 300, 1000, 500, 50)
        params["points"] = st.toggle("Show Point Markers", True)
        params["Y_zero"] = st.toggle("Include zero on Y-axis")
        params["X_grid"] = st.toggle("Show X-axis Grid", True)
        params["Y_grid"] = st.toggle("Show Y-axis Grid", True)
        params["X_inter"] = st.toggle("Interactive X-axis", True)
        params["Y_inter"] = st.toggle("Interactive Y-axis", True)

    plot_history_btn = st.button("Plot History")

    if plot_history_btn:
        try:
            chart = mm.show_history_plot(Y, params, model)
            st.altair_chart(chart, use_container_width=True)
        except (err.NoModelError, err.ModelNotTrainedError) as error:
            st.error(error, icon="❌")
