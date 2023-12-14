from abc import ABC, abstractmethod

import streamlit as st

from mlui.classes.errors import LayerError
from mlui.enums import activations
from mlui.types.classes import (
    BatchNormalizationParams,
    DenseParams,
    DropoutParams,
    InputParams,
    Layer,
    LayerConnection,
    LayerParams,
)


class LayerWidget(ABC):
    """Base class for a layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        self._layers = layers
        self._layer_name = st.text_input(
            "Enter layer's name:", max_chars=50, placeholder="Enter the name"
        )

    @abstractmethod
    def get_connection(self) -> LayerConnection:
        """Get the layer's connection.

        Returns
        -------
        Layer, list of Layer
            Connection layer(s) from the given dictionary.
        """

    @property
    @abstractmethod
    def params(self) -> LayerParams:
        """Layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """


class Input(LayerWidget):
    """Input layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        super().__init__(layers)

        self._input_shape = st.number_input(
            "Number of input columns:", value=1, min_value=1, max_value=10_000
        )

    def get_connection(self) -> None:
        """Input layer's connection."""
        return

    @property
    def params(self) -> InputParams:
        """Input layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"name": self._layer_name, "shape": (int(self._input_shape),)}


class Dense(LayerWidget):
    """Dense layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        super().__init__(layers)

        self._units_num = st.number_input(
            "Number of units:", value=1, min_value=1, max_value=10_000
        )
        activation_options = activations.classes.keys()
        connect_to_options = self._layers.keys()
        self._activation = st.selectbox("Activation function:", activation_options)
        self._connect_to = st.selectbox("Connect layer to:", connect_to_options)

    def get_connection(self) -> Layer:
        """Dense layer's connection.

        Returns
        -------
        Layer
            Layer's object.

        Raises
        ------
        LayerError
            If no layer is selected to connect to.
        """
        if not self._connect_to:
            raise LayerError("Please, select the connection!")

        return self._layers[self._connect_to]

    @property
    def params(self) -> DenseParams:
        """Dense layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "name": self._layer_name,
            "units": int(self._units_num),
            "activation": activations.classes[self._activation]
            if self._activation
            else activations.classes["Linear"],
        }


class Concatenate(LayerWidget):
    """Concatenate layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        super().__init__(layers)

        options = self._layers.keys()
        self._concatenate = st.multiselect("Select layers (at least 2):", options)

    def get_connection(self) -> list[Layer]:
        """Concatenate layer's connection.

        Returns
        -------
        list of Layer
            List of layers' objects.

        Raises
        ------
        LayerError
            If fewer than 2 layers are selected for concatenation.
        """
        if len(self._concatenate) < 2:
            raise LayerError("Please, select the layers to concatenate!")

        return [self._layers[name] for name in self._concatenate]

    @property
    def params(self) -> LayerParams:
        """Concatenate layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"name": self._layer_name}


class BatchNormalization(LayerWidget):
    """BatchNormalization layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        super().__init__(layers)

        self._momentum = st.number_input(
            "Momentum:", value=0.99, min_value=1e-2, max_value=1.0, step=1e-2
        )
        self._epsilon = st.number_input(
            "Epsilon",
            value=1e-3,
            min_value=1e-4,
            max_value=1e-2,
            step=1e-4,
            format="%e",
        )
        options = self._layers.keys()
        self._connect_to = st.selectbox("Connect layer to:", options)

    def get_connection(self) -> Layer:
        """BatchNormalization layer's connection.

        Returns
        -------
        Layer
            Layer's object.

        Raises
        ------
        LayerError
            If no layer is selected to connect to.
        """
        if not self._connect_to:
            raise LayerError("Please, select the connection!")

        return self._layers[self._connect_to]

    @property
    def params(self) -> BatchNormalizationParams:
        """BatchNormalization layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "name": self._layer_name,
            "momentum": float(self._momentum),
            "epsilon": float(self._epsilon),
        }


class Dropout(LayerWidget):
    """Dropout layer's widget."""

    def __init__(self, layers: dict[str, Layer]) -> None:
        """
        Parameters
        ----------
        layers : dict of Layer
            Dictionary of layers to connect to.
        """
        super().__init__(layers)

        self._rate = st.number_input(
            "Rate:", value=0.2, min_value=1e-2, max_value=0.99, step=1e-2
        )
        options = self._layers.keys()
        self._connect_to = st.selectbox("Connect layer to:", options)

    def get_connection(self) -> Layer:
        """Dropout layer's connection.

        Returns
        -------
        Layer
            Layer's object.

        Raises
        ------
        LayerError
            If no layer is selected to connect to.
        """
        if not self._connect_to:
            raise LayerError("Please, select the connection!")

        return self._layers[self._connect_to]

    @property
    def params(self) -> DropoutParams:
        """Dropout layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"name": self._layer_name, "rate": float(self._rate)}
