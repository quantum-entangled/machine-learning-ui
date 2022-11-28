from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Data:
    """Data container."""

    file: pd.DataFrame = pd.DataFrame()
    columns: list[str] = field(default_factory=list)
    num_columns_per_layer: dict[str, int] = field(default_factory=dict)
