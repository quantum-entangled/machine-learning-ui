from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def compile_model(self) -> None:
        ...

    @property
    def optimizer(self) -> Any:
        ...

    @property
    def losses(self) -> dict[str, Any]:
        ...


class CompileModel(iw.VBox):

    name = "Compile Model"

    def __init__(self, model_manager: ModelManager, **kwargs):
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.compile_button = iw.Button(description="Compile Model")
        self.compile_status = iw.Output()

        # Callbacks
        self.compile_button.on_click(self._on_compile_button_clicked)

        super().__init__(children=[self.compile_button, self.compile_status])

    def _on_compile_button_clicked(self, _) -> None:
        self.compile_status.clear_output(wait=True)

        with self.compile_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.model_manager.optimizer:
                print(Error.NO_OPTIMIZER)
                return

            if not all(self.model_manager.losses.values()):
                print(Error.NO_LOSS)
                return

            self.model_manager.compile_model()

            print(Success.MODEL_COMPILED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.compile_status.clear_output()

    def _on_optimizer_selected(self) -> None:
        """Callback for selecting the optimizer."""
        self.compile_status.clear_output()

    def _on_losses_selected(self) -> None:
        """Callback for selecting the losses."""
        self.compile_status.clear_output()
