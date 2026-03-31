
# Installation

`soerp3` can be installed from `pypi`, `conda-forge`, and `git`.

## [PyPI](https://pypi.org/project/soerp3)

For using the PyPI package in your project, you can update your configuration file by adding following snippet.

=== "pyproject.toml"

	```toml
	[project.dependencies]
	soerp3 = "*" # (1)!
	```

	1. Specifying a version is recommended

=== "requirements.txt"

	```
	soerp3>=0.1.0
	```

### pip

=== "Installation for user"

	```bash
	pip install --upgrade --user soerp3 # (1)!
	```

	1. You may need to use `pip3` instead of `pip` depending on your python installation.

=== "Installation in virtual environment"

	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install --require-virtualenv --upgrade soerp3 # (1)!
	```

	1. You may need to use `pip3` instead of `pip` depending on your python installation.

	!!! note
		Command to activate the virtual env depends on your platform and shell. [More info](https://docs.python.org/3/library/venv.html#how-venvs-work)

### pipenv

	pipenv install soerp3

### uv

=== "Adding to uv project"

	```bash
	uv add soerp3
	uv sync
	```

=== "Installing to uv environment"

	```bash
	uv venv
	uv pip install soerp3
	```

### poetry

```bash
poetry add soerp3
```

### pdm

```bash
pdm add soerp3
```

### hatch

```bash
hatch add soerp3
```

## [conda-forge](https://anaconda.org/conda-forge/soerp3)

You can update your environment spec file by adding following snippets.

```yaml title="environment.yml"
channels:
  - conda-forge
dependencies:
  - pip
  - pip:
      - soerp3 # (1)!
```

1. Specifying a version is recommended

Installation can be done using the updated environment spec file.

=== "conda"
	```bash
	conda env update --file environment.yml
	```
=== "micromamba"
	```bash
	# Installation

	soerp3 can be installed from PyPI, conda-forge, or directly from git.

	## [PyPI](https://pypi.org/project/soerp3)

	=== "pyproject.toml"
		```toml
		[project.dependencies]
		soerp3 = "*" # (1)!
		```
		1. Specifying a version is recommended

	=== "requirements.txt"
		```
		soerp3>=0.1.0
		```

	### pip

	=== "Installation for user"
		```bash
		pip install --upgrade --user soerp3 # (1)!
		```
		1. You may need to use `pip3` instead of `pip` depending on your python installation.

	=== "Installation in virtual environment"
		```bash
		python -m venv .venv
		source .venv/bin/activate
		pip install --require-virtualenv --upgrade soerp3 # (1)!
		```
		1. You may need to use `pip3` instead of `pip` depending on your python installation.

		!!! note
			Command to activate the virtual env depends on your platform and shell. [More info](https://docs.python.org/3/library/venv.html#how-venvs-work)

	### pipenv
		pipenv install soerp3

	### uv
	=== "Adding to uv project"
		```bash
		uv add soerp3
		uv sync
		```
	=== "Installing to uv environment"
		```bash
		uv venv
		uv pip install soerp3
		```

	### poetry
	```bash
	poetry add soerp3
	```

	### pdm
	```bash
	pdm add soerp3
	```

	### hatch
	```bash
	hatch add soerp3
	```

	## [conda-forge](https://anaconda.org/conda-forge/soerp3)

	```yaml title="environment.yml"
	channels:
	  - conda-forge
	dependencies:
	  - soerp3 # (1)!
	```
	1. Specifying a version is recommended

	=== "conda"
		```bash
		conda env update --file environment.yml
		```
	=== "micromamba"
		```bash
		micromamba env update --file environment.yml
		```

	!!! note
		replace `environment.yml` with your actual environment spec file name if it's different.

	=== "conda"
		```bash
		conda install -c conda-forge soerp3
		```
	=== "micromamba"
		```bash
		micromamba install -c conda-forge soerp3
		```

	## [git](https://github.com/eggzec/soerp3)

	```bash
	pip install --upgrade "git+https://github.com/eggzec/soerp3.git#egg=soerp3"
	```

	## Dependencies

	- Python >=3.10
	- [numpy](https://pypi.org/project/numpy)
	- [scipy](https://pypi.org/project/scipy)
	- [matplotlib](https://pypi.org/project/matplotlib) (optional)
