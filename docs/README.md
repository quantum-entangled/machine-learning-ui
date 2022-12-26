# Build Docs

To build **HTML** version of docs with **Sphinx** run either of these commands within a root directory:
```sh
sphinx-build -E -b html docs/src/ docs/build/html
```
```sh
make clean html
```

And then open `docs/build/index.html` via your browser.
