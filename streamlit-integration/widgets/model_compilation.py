import data_classes.model as model_cls
import streamlit as st

import enums.losses as losses
import enums.optimizers as optimizers
import managers.errors as err
import managers.model_manager as mm


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

    layer = str(st.selectbox("Select layer:", list(model.output_layers)))
    loss = str(st.selectbox("Select loss function class:", list(losses.classes)))
    loss_cls = losses.classes[loss]
    set_loss_btn = st.button("Set Loss Function")

    if set_loss_btn:
        try:
            mm.set_loss(layer, loss_cls, model)
            st.success("Loss is set!", icon="✅")
        except (err.NoModelError, err.NoOutputLayersError) as error:
            st.error(error, icon="❌")
