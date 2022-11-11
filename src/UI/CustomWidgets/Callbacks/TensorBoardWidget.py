from typing import Any
import ipywidgets as iw


class TensorBoardWidget(iw.VBox):

    name = "TensorBoard Callback"

    def __init__(self, **kwargs):
        self.write_graph = iw.Checkbox(value=True, description="Write graph")
        self.write_images = iw.Checkbox(value=False, description="Write images")

        super().__init__(children=[self.write_graph, self.write_images], **kwargs)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "log_dir": "../db/Logs/graph",
            "write_graph": self.write_graph.value,
            "write_images": self.write_images.value,
        }
