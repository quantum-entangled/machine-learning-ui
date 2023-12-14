import streamlit as st

from mlui.classes.data import Data
from mlui.classes.errors import SetError
from mlui.classes.model import Model
from mlui.enums import callbacks


def set_features_ui(data: Data, model: Model) -> None:
    """Generate the UI for setting the data columns
    as features for the model's input and output layers.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Set Input/Output Features")
    st.markdown(
        "Here, you can set the input and output features for each respective layer. "
        "The changes will take place only after clicking the `Set Features` button. "
        "Please, be aware that the order in which you add the columns is important, "
        "as your future data need to be consistent with the data on which the model "
        "was trained."
    )

    task = st.session_state.get("task")

    if task != "Predict":
        side = st.selectbox("Select layer's type:", ("Input", "Output"))
    else:
        side = st.selectbox("Select layer's type:", ("Input",))

    if side == "Input":
        at = "input"
        layers = model.inputs
        shapes = model.input_shape
    else:
        at = "output"
        layers = model.outputs
        shapes = model.output_shape

    def set_features() -> None:
        try:
            model.set_features(layer, columns, at)
            data.set_unused_columns(options, columns)
            st.toast("Features are set!", icon="✅")
        except SetError as error:
            st.toast(error, icon="❌")

    layer = str(st.selectbox("Select layer:", layers))
    features = model.get_features(layer, at)
    options = data.get_unused_columns()
    options.extend(features)
    columns = st.multiselect(
        "Select columns (order is important):",
        options,
        features,
        max_selections=shapes[layer],
    )

    st.button("Set Features", on_click=set_features)


def set_callbacks_ui(model: Model) -> None:
    """Generate the UI for setting the model's callbacks.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Callbacks")

    options = callbacks.classes.keys()
    entity = str(st.selectbox("Select callback's class:", options))

    with st.expander("Callback's Parameters"):
        widget = callbacks.widgets[entity]()

    def manage_callback() -> None:
        if not callback_is_set:
            try:
                params = widget.params

                model.set_callback(entity, params)
                st.toast("Callback is set!", icon="✅")
            except SetError as error:
                st.toast(error, icon="❌")
        elif callback_is_set:
            model.delete_callback(entity)
            st.toast("Callback is deleted!", icon="✅")

    callback_is_set = model.get_callback(entity)
    label = "Set Callback" if not callback_is_set else "Delete Callback"

    st.button(label, on_click=manage_callback)
