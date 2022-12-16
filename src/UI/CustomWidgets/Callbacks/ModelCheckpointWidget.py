from typing import Any
import ipywidgets as iw


class ModelCheckpointWidget(iw.VBox):

    name = "ModelCheckpoint Callback"

    @property
    def params(self) -> dict[str, Any]:
        return {"filepath": "db/Logs/checkpoints/model.ckpt"}
