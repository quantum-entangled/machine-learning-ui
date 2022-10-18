from ipydatagrid import DataGrid
from IPython.display import clear_output, display
from ipywidgets import Button, Output, VBox


class DataGridWidget(VBox):
    def __init__(self, file_uploader):
        self.file_uploader = file_uploader
        self.show_grid_button = Button(description="Show Data Grid")
        self.grid_output = Output()
        self.widget_children = [self.show_grid_button, self.grid_output]

        self.show_grid_button.on_click(self.__show_data_grid)

        super().__init__(children=self.widget_children)

    def __show_data_grid(self, *args):
        if self.file_uploader.file is not None:
            with self.grid_output:
                clear_output()
                display(
                    DataGrid(
                        dataframe=self.file_uploader.file,
                    )
                )
        else:
            with self.grid_output:
                clear_output()
                print("Please, upload the file first!")
