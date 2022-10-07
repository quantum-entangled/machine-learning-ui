from ipywidgets import FileUpload


class UploadFile(FileUpload):
    def __init__(self):
        self.accepted_file_extansions = ".txt"
        self.multiple_files_upload = False

        super().__init__(
            accept=self.accepted_file_extansions, multiple=self.multiple_files_upload
        )
