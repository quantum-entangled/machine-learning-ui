import streamlit as st

import mlui.data_classes.model as model_cls
from mlui.enums import losses, metrics, optimizers

from ..managers import errors as err
from ..managers import model_manager as mm


def set_optimizer_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model's optimizer.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Optimizer")

    optimizer = str(st.selectbox("Select optimizer class:", list(optimizers.classes)))
    optimizer_cls = optimizers.classes[optimizer]
    optimizer_widget = optimizers.widgets[optimizer]()
    optimizer_params = optimizer_widget.params
    set_optimizer_btn = st.button("Set Optimizer")

    if set_optimizer_btn:
        try:
            mm.set_optimizer(optimizer_cls, optimizer_params, model)
            st.success("Optimizer is set!", icon="✅")
        except (err.NoModelError, err.NoOutputLayersError) as error:
            st.error(error, icon="❌")


def set_loss_functions_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model's loss functions.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Loss Functions")

    layer = str(st.selectbox("Select layer:", list(model.output_layers), key="loss"))
    loss = str(st.selectbox("Select loss function class:", list(losses.classes)))
    loss_cls = losses.classes[loss]
    set_loss_btn = st.button("Set Loss Function")

    if set_loss_btn:
        try:
            mm.set_loss(layer, loss_cls, model)
            st.success("Loss is set!", icon="✅")
        except (err.NoModelError, err.NoOutputLayersError) as error:
            st.error(error, icon="❌")


def set_metrics_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model's metrics.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Metrics")

    layer = str(st.selectbox("Select layer:", list(model.output_layers), key="metric"))
    layer_metrics = list(model.metrics[layer])
    selected_metrics = st.multiselect(
        "Select metrics for this layer:", list(metrics.classes), layer_metrics
    )
    set_metrics_btn = st.button("Set Metrics")

    if set_metrics_btn:
        try:
            metrics_to_set = {
                name: metrics.classes[name]() for name in selected_metrics
            }

            mm.set_metrics(layer, metrics_to_set, model)
            st.success("Metrics are set!", icon="✅")
        except (err.NoModelError, err.NoOutputLayersError) as error:
            st.error(error, icon="❌")


def compile_model_ui(model: model_cls.Model) -> None:
    """Generate UI for compiling the model.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Compile Model")

    compile_model_btn = st.button("Compile Model")

    if compile_model_btn:
        try:
            mm.compile_model(model)
            st.success("Model is compiled!", icon="✅")
        except (err.NoModelError, err.NoOptimizerError, err.NoLossError) as error:
            st.error(error, icon="❌")
