# Example Python Package

This is a bare bones example of a Python package.

## Installation

Using [Mamba](https://mamba.readthedocs.io/en/latest/):

* Create a new Mamba environment

```bash
mamba create -n example_environment python=3.12
```

* Activate the Mamba environment

```bash
mamba activate example_environment
```

* Navigate into the top level folder and pip install the package.

```bash
pip install -e .
```

* Open a Python REPL and try running an example to test the installation

```python
import magicc
magicc.fly()
```

## Developer Dependencies

To install optional packages to help with development, run `pip` with the `.[dev]` option:

```bash
pip install -e ".[dev]"
```

This will install (among other things) [Pytest](https://docs.pytest.org/en/8.2.x/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).


## Writing Documentation

This repo has [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) set up. You can read more at the [official documentation](https://squidfunk.github.io/mkdocs-material/getting-started/).

To launch the documentation locally in a web browser:

* Install the developer dependencies as per above
* Activate the Mamba environment and navigate to the top level folder
* Run the following:

```bash
mkdocs serve
```

* Open the link that appears in the terminal.

Changes to the files in `docs` are reflected live as you save individual documents.