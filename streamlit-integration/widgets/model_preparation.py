import data_classes.model as model_cls
import streamlit as st

import enums.layers as layers
import managers.errors as err
import managers.model_manager as mm


def create_model_ui(model: model_cls.Model) -> None:
    """Generate UI for creating a new model.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Create Model")

    model_name = st.text_input(
        "Model name:", max_chars=50, placeholder="Enter Model Name"
    )
    create_model_btn = st.button("Create Model")

    if create_model_btn:
        try:
            mm.create_model(model_name, model)
            st.success("Model is created!", icon="✅")
        except (err.NoModelNameError, err.SameModelNameError) as error:
            st.error(error, icon="❌")


def add_layers_ui(model: model_cls.Model) -> None:
    """Generate UI for managing model's layers.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Add Layers")

    layer_type = str(st.selectbox("Select layer type:", list(layers.instances)))
    layer_instance = layers.instances[layer_type]
    layer_widget = layers.widgets[layer_type](model=model)
    layer_params = layer_widget.params
    layer_connection = layer_widget.connection
    add_layer_btn = st.button("Add Layer")

    if add_layer_btn:
        try:
            mm.add_layer(layer_instance, layer_params, layer_connection, model)
            st.success("Layer is added!", icon="✅")
        except (
            err.NoModelError,
            err.NoLayerNameError,
            err.SameLayerNameError,
            err.NoConnectionError,
        ) as error:
            st.error(error, icon="❌")


def set_outputs_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model ouputs.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Outputs")

    outputs = st.multiselect(
        "Select outputs:", list(set(model.layers) - set(model.input_layers))
    )
    set_outputs_btn = st.button("Set Outputs")

    if set_outputs_btn:
        try:
            mm.set_outputs(outputs, model)
            st.success("Outputs are set!", icon="✅")
        except (err.NoModelError, err.NoOutputsSelectedError) as error:
            st.error(error, icon="❌")
