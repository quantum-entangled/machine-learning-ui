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
    Y = st.multiselect("Select Y column(s):", opts)
    df_len = len(model.training_history)
    X_L = -0.5 * df_len
    X_R = df_len + 0.5 * df_len
    Y_min = model.training_history.loc[:, Y].min(axis=1).min(axis=0)
    Y_max = model.training_history.loc[:, Y].max(axis=1).max(axis=0)
    Y_L = Y_min - 0.5 * Y_max
    Y_R = Y_max + 0.5 * Y_max
    schemes = ("set1", "set2", "set3")
    legend_ors = (
        "left",
        "right",
        "top",
        "bottom",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    )
    legend_dirs = ("vertical", "horizontal")
    title_ph = "No title if None"

    with st.expander("Chart's Parameters"):
        scheme = st.selectbox("Select color scheme:", schemes, 0)
        X_ticks = st.number_input("Number of X-axis ticks:", 0, int(X_R), df_len, 1)
        Y_ticks = st.number_input("Number of Y-axis ticks:", 0, int(X_R), df_len, 1)
        X_l_lim = st.number_input("X-axis left border:", X_L, X_R, 0.5, 1.0)
        X_r_lim = st.number_input("X-axis right border:", X_L, X_R, df_len + 0.5, 1.0)
        Y_l_lim = st.number_input(
            "Y-axis left border:", Y_L, Y_R, Y_min - 0.05 * Y_min, 1.0
        )
        Y_r_lim = st.number_input(
            "Y-axis right border:", Y_L, Y_R, Y_max + 0.05 * Y_max, 1.0
        )
        X_title = st.text_input("X-axis title:", "Epoch", 30, placeholder=title_ph)
        Y_title = st.text_input("Y-axis title:", "Value", 30, placeholder=title_ph)
        legend_or = st.selectbox("Legend orientation:", legend_ors, 3)
        legend_dir = st.selectbox("Legend direction:", legend_dirs, 1)
        legend_title = st.text_input("Legend title:", None, 30, placeholder=title_ph)
        height = st.number_input("Plot height:", 300, 1000, 500, 50)
        points = st.toggle("Show Point Markers", True)
        Y_zero = st.toggle("Include zero on Y-axis")
        X_grid = st.toggle("Show X-axis Grid", True)
        Y_grid = st.toggle("Show Y-axis Grid", True)
        X_inter = st.toggle("Interactive X-axis", True)
        Y_inter = st.toggle("Interactive Y-axis", True)

    plot_history_btn = st.button("Plot History")

    if plot_history_btn:
        try:
            chart = mm.show_history_plot(
                Y,
                {
                    "scheme": scheme if scheme else "set1",
                    "X_ticks": X_ticks,
                    "Y_ticks": Y_ticks,
                    "X_l_lim": X_l_lim,
                    "X_r_lim": X_r_lim,
                    "Y_l_lim": Y_l_lim if Y_l_lim else Y_min,
                    "Y_r_lim": Y_r_lim if Y_r_lim else Y_max,
                    "X_title": X_title,
                    "Y_title": Y_title,
                    "legend_or": legend_or if legend_or else "bottom",
                    "legend_dir": legend_dir if legend_dir else "horizontal",
                    "legend_title": legend_title,
                    "height": height,
                    "points": points,
                    "Y_zero": Y_zero,
                    "X_grid": X_grid,
                    "Y_grid": Y_grid,
                    "X_inter": X_inter,
                    "Y_inter": Y_inter,
                },
                model,
            )
            st.altair_chart(chart, use_container_width=True)
        except (err.NoModelError, err.ModelNotTrainedError) as error:
            st.error(error, icon="❌")
