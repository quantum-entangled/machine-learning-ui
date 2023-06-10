import data_classes.data as data_cls
import data_classes.model as model_cls
import streamlit as st

import managers.data_manager as dm
import managers.errors as err


def set_columns_ui(data: data_cls.Data, model: model_cls.Model) -> None:
    """Generate UI for setting the data columns for the model's input and output layers.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    st.header("Set Input/Output Columns")
    st.markdown(
        "Here, you can set the input and output columns for each respective layer. "
        "The changes will take place only after clicking the `Set Columns` button. "
        "Please, be aware that the order in which you add the columns is important, "
        "as your future data need to be consistent with the data on which the model "
        "was trained."
    )

    layer_type = st.selectbox("Select layer type:", ("Input", "Output"))

    if layer_type == "Input":
        layers = model.input_layers
        shapes = model.input_shapes
    else:
        layers = model.output_layers
        shapes = model.output_shapes

    layer = st.selectbox("Select layer:", list(layers))

    if layer_type and layer:
        layer_columns = dm.get_layer_columns(layer_type, layer, data)
        options = data.available_columns + layer_columns
        columns = st.multiselect(
            "Select columns for this layer:", options, layer_columns
        )

        if shapes.get(layer):
            st.write(f"Layer fullness: {len(columns)}/{shapes[layer]}")
            add_columns_btn = st.button("Set Columns")

            if add_columns_btn:
                try:
                    dm.set_columns(layer_type, layer, columns, data, model)
                    st.success("Columns are set!", icon="✅")
                except (err.NoColumnsSelectedError, err.LayerOverfilledError) as error:
                    st.error(error, icon="❌")
