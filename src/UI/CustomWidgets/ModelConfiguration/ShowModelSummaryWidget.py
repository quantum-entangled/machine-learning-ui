from typing import Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def show_model_summary(self) -> None:
        ...


class ShowModelSummaryWidget(iw.VBox):
    """Show model summary."""

    name = "Show Model Summary"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.show_summary_button = iw.Button(description="Show Summary")
        self.summary = iw.Output()

        # Callbacks
        self.show_summary_button.on_click(self._on_show_summary_button_clicked)

        super().__init__(children=[self.show_summary_button, self.summary])

    def _on_show_summary_button_clicked(self, _) -> None:
        """Callback for show summary button."""
        self.summary.clear_output(wait=True)

        with self.summary:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            self.model_manager.show_model_summary()

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.summary.clear_output()
