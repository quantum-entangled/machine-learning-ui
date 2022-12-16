from typing import Any, Protocol

from src.Managers import DataManager, ModelManager


class ChildWidget(Protocol):
    """Child widget protocol class."""

    name: str

    def __call__(self, *args, **kwargs) -> Any:
        ...


class Widget(Protocol):
    """Main widget protocol class."""

    children: Any
    titles: Any

    def __call__(self, *args, **kwargs) -> Any:
        ...


class WidgetUI:
    """General class for rendering widgets display data."""

    def __init__(
        self,
        widget: Widget,
        widget_children: list[ChildWidget],
        data_manager: DataManager,
        model_manager: ModelManager,
        **kwargs
    ) -> None:
        """Initialize the UI."""
        # Widgets
        self.ui = widget(
            children=[
                child_widget(
                    data_manager=data_manager, model_manager=model_manager, **kwargs
                )
                for child_widget in widget_children
            ],
            titles=[child_widget.name for child_widget in widget_children],
        )

        # Callbacks
        data_manager.observers.extend(self.ui.children)
        model_manager.observers.extend(self.ui.children)
