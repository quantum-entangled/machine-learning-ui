import streamlit as st

from mlui.classes.errors import CreateError, LayerError, SetError
from mlui.classes.model import CreatedModel
from mlui.enums import layers


def set_name_ui(model: CreatedModel) -> None:
    """Generate the UI for setting the model's name.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Name")

    name = st.text_input(
        "Enter model's name:", max_chars=50, placeholder="Enter the name"
    )

    def set_name() -> None:
        try:
            model.set_name(name)
            st.toast("Name is set!", icon="✅")
        except SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Name", on_click=set_name)


def set_layers_ui(model: CreatedModel) -> None:
    """Generate the UI for setting the model's layers.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Layers")

    options = layers.classes.keys()
    entity = str(st.selectbox("Select layer's class:", options))

    with st.expander("Layer's Parameters"):
        widget = layers.widgets[entity](model.layers)

    def set_layer() -> None:
        try:
            params = widget.params
            connection = widget.get_connection()

            model.set_layer(entity, params, connection)
            st.toast("Layer is set!", icon="✅")
        except (LayerError, SetError) as error:
            st.toast(error, icon="❌")

    st.button("Set Layer", on_click=set_layer)


def set_outputs_ui(model: CreatedModel) -> None:
    """Generate the UI for setting the model's outputs.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Outputs")

    options = model.layers.keys() - model.inputs
    outputs = st.multiselect("Select outputs:", options)

    def set_outputs() -> None:
        try:
            model.set_outputs(outputs)
            st.toast("Outputs are set!", icon="✅")
        except SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Outputs", on_click=set_outputs)


def create_model_ui(model: CreatedModel) -> None:
    """Generate the UI for building the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Create Model")

    def build_model() -> None:
        try:
            model.create()
            st.toast("Model is created!", icon="✅")
        except CreateError as error:
            st.toast(error, icon="❌")

    st.button("Create Model", on_click=build_model)
