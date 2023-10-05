import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.data_manager as dm
import mlui.managers.errors as err


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

        with st.form("Columns Form"):
            columns = st.multiselect(
                "Select columns for this layer (order is important):",
                options,
                layer_columns,
                max_selections=shapes[layer],
            )
            add_columns_btn = st.form_submit_button("Set Columns")

            if add_columns_btn:
                try:
                    dm.set_columns(layer_type, layer, columns, data, model)
                    st.success("Columns are set!", icon="✅")
                except (err.NoColumnsSelectedError, err.LayerOverfilledError) as error:
                    st.error(error, icon="❌")


def split_data_ui(data: data_cls.Data, model: model_cls.Model) -> None:
    """Generate UI for splitting the data into training and test sets.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    st.header("Split Data")

    test_size = float(
        st.slider(
            "Select the percent of test data:",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
        )
    )
    split_data_btn = st.button("Split Data")

    if split_data_btn:
        try:
            dm.split_data(test_size, data, model)
            st.success("Dataset is split!", icon="✅")
        except (
            err.InputsUnderfilledError,
            err.OutputsUnderfilledError,
            err.IncorrectTestDataPercentage,
        ) as error:
            st.error(error, icon="❌")
