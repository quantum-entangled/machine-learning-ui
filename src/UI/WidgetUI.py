from typing import Any, Mapping, Protocol, Sequence


class ChildWidget(Protocol):
    "Child widget protocol class."

    name: str

    def __call__(self, *args, **kwargs) -> Any:
        ...


class Widget(Protocol):
    "Main widget protocol class."
    children: Any
    titles: Any
    selected_index: Any

    def __call__(self, *args, **kwargs) -> Any:
        ...

    def observe(self, names: Any, *args, **kwargs) -> Any:
        ...


class WidgetUI:
    """General class for rendering widgets display data."""

    def __init__(
        self,
        widget: Widget,
        widget_children: Sequence[ChildWidget],
        widget_params: Mapping[str, Any],
        **kwargs
    ) -> None:
        """Initialize the UI."""
        self.ui = widget(
            children=[
                child_widget(**widget_params) for child_widget in widget_children
            ],
            titles=[child_widget.name for child_widget in widget_children],
            **kwargs
        )
        self.ui.observe(self._on_widget_selected_index_change, names="selected_index")

    def _on_widget_selected_index_change(self, _) -> None:
        for child_widget in self.ui.children:
            if callable(getattr(child_widget, "_on_widget_state_change", None)):
                child_widget._on_widget_state_change()
