from typing import Any
import ipywidgets as iw


class CSVLoggerWidget(iw.VBox):

    name = "CSVLogger Callback"

    @property
    def params(self) -> dict[str, Any]:
        return {"filename": "db/Logs/csv/log.csv"}
