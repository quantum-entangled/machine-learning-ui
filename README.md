[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quantum-entangled/machine-learning-ui/HEAD?urlpath=voila%2Frender%2Fsrc%2Fmain.ipynb)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://quantum-entangled.github.io/machine-learning-ui/index.html)

The app provides User Interface based on Jupyter's [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) for managing basic Machine Learning workflows. You can launch it using Binder badge above or installing app locally.

The app was originally developed as part of research project, funded by **St. Petersburg State University**, to help researchers solve problems of non-equilibrium gas dynamics using machine learning methods.

## Local Installation Guide

First of all, pull the repository to your local machine.

- **(Git is installed)** Run `git clone https://github.com/quantum-entangled/machine-learning-ui`.
- **(Git is not installed)** [Git Installation Guide](https://github.com/git-guides/install-git) / [Git Bash](https://git-scm.com/downloads).
- **(Git is not installed and you don't want to do that)** Just download the ZIP-folder with source code via green `Code` button in the upper right corner of the page.

Next, use either **conda** or built-in **venv**.

### conda:

1) Install either **Miniconda** or **Anaconda** via this [Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2) Open **Anaconda Prompt** and `cd` to the repository directory on your machine. Useful guide if you're not familiar with `cd`: [Video](https://www.youtube.com/watch?v=KNjzcJhUwuA).

3) Run `conda env create -f .binder/environment.yml`. Wait for the installation.

4) Run `conda activate machine-learning-ui`.

5) Run `jupyter notebook main.ipynb`. 

6) Activate **Voilà** via the status bar button to automatically run and hide all input cells.

7) Repeat steps 2, 4-6 every time you start the app.

### venv:

Change `/` to `\` everywhere except URLs, if you're using Windows.

1) Install **Python 3.10** for your system: [Download](https://www.python.org/downloads/release/python-3108/).
    - Make sure your system meets all the requirements (for Windows users: Win8 and higher).
    - Make sure to toggle "Add to PATH" checkbox in the installation menu if you have some.
    - **(Optional)** Delete other Python versions installed on your system or make sure to launch Jupyter via Python 3.10.

2) Open your terminal/cmd (not **conda** or similar, just system built-in) and `cd` to the repository directory on your machine. Useful guide if you're not familiar with `cd`: [Video](https://www.youtube.com/watch?v=KNjzcJhUwuA).

3) Run `python -m venv venv`.

4) Run `venv/Scripts/activate.bat` or `source venv/Scripts/activate` to activate your environment, if **(venv)** badge has not appeared.

5) Run `pip install -r .binder/requirements.txt`. Wait for the installation.

6) Run `jupyter notebook main.ipynb`. 

7) Activate **Voilà** via the status bar button to automatically run and hide all input cells.

8) Repeat steps 2, 4, 6 and 7 every time you start the app.
