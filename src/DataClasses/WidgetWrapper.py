from dataclasses import dataclass
from typing import Any


@dataclass
class WidgetWrapper:
    """Wrapper for different instances and widgets."""

    instance: Any = None
    widget: Any = None
