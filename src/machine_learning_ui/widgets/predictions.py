import data_classes.data as data_cls
import data_classes.model as model_cls
import managers.errors as err
import managers.model_manager as mm
import streamlit as st


def make_predictions_ui(data: data_cls.Data, model: model_cls.Model) -> None:
    """Generate UI for making the model predictions.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    st.header("Make Predictions")

    batch_size = int(
        st.number_input("Batch size:", min_value=1, max_value=1024, value=32, step=1)
    )
    make_predictions_btn = st.button("Make Predictions")

    if make_predictions_btn:
        try:
            predictions = mm.make_predictions(batch_size, data, model)
            st.write("Predictions:", predictions)
            st.success("Predictions are completed!", icon="✅")
        except (
            err.NoModelError,
            err.InputsUnderfilledError,
            err.ModelNotCompiledError,
        ) as error:
            st.error(error, icon="❌")
