from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    ...


class SetModelOutputsWidget(iw.VBox):
    """Widget to add and pop model layers."""

    name = "Set Model Outputs"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager
