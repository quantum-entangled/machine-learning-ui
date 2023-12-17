import streamlit as st

import mlui.classes.data as data
import mlui.classes.errors as errors
import mlui.classes.model as model


def evaluate_model_ui(data: data.Data, model: model.UploadedModel) -> None:
    """Generate the UI for evaluating the model.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Evaluate Model")

    batch_size = st.number_input(
        "Batch size:", min_value=1, max_value=1024, value=32, step=1
    )
    evaluate_model_btn = st.button("Evaluate Model")

    if evaluate_model_btn:
        with st.status("Evaluation Results"):
            try:
                df = data.dataframe
                results = model.evaluate(df, int(batch_size))

                st.subheader("Tracked metrics and losses")
                st.dataframe(results, hide_index=True)
                st.toast("Evaluation is completed!", icon="✅")
            except errors.ModelError as error:
                st.toast(error, icon="❌")
