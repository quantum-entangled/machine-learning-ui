[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quantum-entangled/machine-learning-ui/v0.1.1?urlpath=voila%2Frender%2Fmain.ipynb)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://quantum-entangled.github.io/machine-learning-ui/index.html)

The app provides User Interface based on Jupyter's [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for managing basic Machine Learning workflows. You can launch it using Binder badge above or installing app locally.

The app was originally developed as part of research project, funded by **St. Petersburg State University**, to help researchers solve problems of non-equilibrium gas dynamics using machine learning methods.


## Installing via Docker (Recommended)

### For Usage

1. Install Docker for your system by following the official guide: [Docker Installation](https://docs.docker.com/engine/install/).
1. Open Docker Desktop, use the search bar to find `ivanshalamov/machine-learning-ui`, and pull the `latest` image.
1. In the "Images" tab, click the `Run` button for the pulled image in the "Actions" column. Enter a name for the container (e.g., `machine-learning-ui-app`) in the "Container name" section. Specify an available port (e.g., `8501`) in the "Ports" section, then run the image.
1. Navigate to the "Containers" tab and click the hyperlink in the "Ports" column to access the app.
1. To stop the app, go to the "Containers" tab again and click `Stop` in the "Actions" column. You can easily restart it later from there without any additional steps.

### For Development
1. Install Docker for your system by following the official guide: [Docker Installation](https://docs.docker.com/engine/install/).
1. Clone the repository to your local machine and navigate to its root folder using `cd`.
1. In the terminal, run `docker compose up`. Building the image might take a while. The app will run at `localhost:8501`.
    > To run the container in detached mode, use the `-d` flag, allowing you to use your current terminal.
1. To stop the app, run `docker compose stop` or press `Ctrl+C`. You can restart it with the `docker compose up` command.
    > In development mode, changes in your local `src`, `docs`, and `tests` folders, as well as Poetry files, are immediately reflected in the container without needing a restart.


## Installing via Poetry
1. Install Poetry for your system by following the official guide: [Poetry Installation](https://python-poetry.org/docs/#installation). Remember to add Poetry to your Path.
1. Clone the repository to your local machine and navigate to its root folder using `cd`.
1. In the terminal, run `poetry install` for basic usage or `poetry install --with docs,tests` for development (comma-separated without spaces). On Windows, include `windows` in the `--with` flag; this is necessary for TensorFlow to function correctly.
1. Start the app by running `streamlit run src/mlui/''$'\360\237\217\240''_Home.py'`. Access the app by navigating to the provided URL.
1. To stop the app, press `Ctrl+C` in the terminal. You can restart it using the previous step.
