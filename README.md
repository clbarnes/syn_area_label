# syn_area_label

For the latest version, see here: https://github.com/clbarnes/syn_area_label

Preparing data for labelling synaptic areas using CATMAID annotations

Originally created using
[cookiecutter](https://github.com/cookiecutter/cookiecutter) and
[clbarnes/python-template-sci](https://github.com/clbarnes/python-template-sci).

## Usage

First, ensure you're working in a virtual environment:

```sh
# create a virtual environment if you don't have one
python -m venv --prompt syn_area_label venv

# activate it
source venv/bin/activate
```

Install this package (in editable mode, so that local changes are picked up immediately),
and dependencies listed in [`requirements.txt`](./requirements.txt),
using `pip install .`.
If you are developing the package, instead use `pip install --editable ".[dev]"`.

Shared utilities are in [`src/syn_area_label/`](./src/syn_area_label).
Scripts using those utilities are in [`scripts/`](./scripts).

### Data

Data files should be kept to the [`data/`](./data) directory,
or the path can be set using the `SYN_AREA_LABEL_DATA` environment variable
(see [`data/README.md`](./data/README.md) for more details).

### Containerisation

This project can be containerised with [apptainer](https://apptainer.org/docs/user/main/quick_start.html) (formerly called Singularity)
(bundling it with a python environment and full OS) on linux,
so that it can be run on any system with apptainer installed.

Just run `make container` (requires sudo).

The python files are installed in the container at `/project`.
To improve container size and flexibility, the `./data/` directory is not included.
To improve flexibility and security, the `./credentials/` directory is not included.

You can [bind mount](https://apptainer.org/docs/user/main/bind_paths_and_mounts.html) these directories inside the container at runtime:

```sh
# Find the data path your environment is using, defaulting to the local ./data
DATA_PATH="${SYN_AREA_LABEL_DATA:-$(pwd)/data}"
CREDS_PATH="$(pwd)/credentials"

# Execute the command `/bin/bash` (i.e. get a terminal inside the container),
# mounting the data directory and credentials you're already using.
# Container file (.sif) must already be built
apptainer exec \
    --bind "$DATA_PATH:/project/data" \
    --bind "$CREDS_PATH:/project/credentials" \
    syn_area_label.sif /bin/bash
```

## Development

This layout is designed to encourage a firm division between general utilities and the scripts.
The scripts should be fairly minimal;
basically just feeding fixed arguments to functions and classes from your utilities.

This encourages code to be re-usable and testable, and keeps your final product readable,
and makes it easier to export useful functions if you need them elsewhere.

More recommendations can be found [here](https://gitlab.com/cardonazlaticlabs/data-policy/-/blob/master/GUIDELINES.md).

### Documentation

Use descriptive variable and function names: long names are not a problem in editors with autocomplete,
where a script full of `a`, `b`, `i`, `j`, `foo` is a pain to read for the next person - who will probably be you.

Where possible, add a [docstring](https://realpython.com/defining-your-own-python-function/#docstrings)
to modules and functions to explain what they do,
and in particular what type of data they take in and put out.
Trying to remember or infer what a function needs and produces is one of the biggest wastes of time in working with scripts.

Using [type hints](https://realpython.com/lessons/type-hinting/) will also allow automated tools to catch certain types of error.

### Credentials

Passwords, API tokens etc. **MUST NOT** be tracked with git.
Files in the `credentials/` directory are ignored by git, so this is a good place to keep them.
The path to this directory can be found in the python variable `syn_area_label.constants.CREDENTIALS_DIR`.

Document what credentials you need for your scripts to function, and what format they need to take,
so that someone with different credentials can still work with your scripts.

### Dependencies

All dependencies should be specified in the requirements file,
ideally with specific versions listed so that your environment is reproducible.
[`pur`](https://pypi.org/project/pur/) can be used to update your requirements file to the latest versions.

Any non-python dependencies should be documented,
and their installation steps should be included in the `%post` section of the Apptainer definition.

### Linting

Linters are powerful tools which try to catch common bugs and errors by analysing your code,
as well as giving hints on how to keep it neat, tidy, and readable.
Formatters make your code uniform and improve readability.

The development install includes standard linters and formatters.
On unix systems, you can run them using `make format` and `make lint`.

### Testing

Unit tests make sure that small utility functions do what you expect them to.
They are generally quick to run and can help prove that the building blocks of your pipeline are sound.

This template uses [pytest](https://docs.pytest.org/).
Any tests you've written run automatically when you push to GitHub.
You can run tests locally with `make test` (on unix).

### Continuous Deployment

This repo uses [GitHub Actions](https://docs.github.com/en/actions) to lint and test your code every time you push to GitHub.

### Versioning

Versioning is useful for ensuring that you or anyone else can reproduce your pipeline exactly as it was at some key moment;
for example if you want to keep working on a repo after submitting a paper based on it but allow people to access the published version easily.

To update the version, simply apply a [git tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) like

```sh
git tag --annotate v2.0 --message "Scripts as used in Myself et al. (2023)"
```

When you `git push` your code up to a remote (e.g. GitHub), include the `--follow-tags` argument to push the tags as well.

This repo uses [`setuptools-scm`](https://pypi.org/project/setuptools-scm/) so that your package version is determined by your git tag.
You can find your package version at `syn_area_label.__version__`.
If you add a new version tag, you may need re-install the package (`pip install -e .`) for the new version to show up here.

### Re. Notebooks

Jupyter (formerly IPython) Notebooks are a nice way of showcasing or publishing small pieces of code.
However, in my experience they are a pain for developing and reproducing pipelines; use plain scripts instead.

VSCode can run notebook-like cells in plain scripts, separated with `#%%`.

## After `cookiecutter`-ing this template

- [ ] Check that the license fits your use case (see [`LICENSE`](./LICENSE)), change or delete otherwise
- [ ] Pick and document a python version you will use
- [ ] Start using git; basic tutorial [here](https://clbarnes.github.io/version-control-tutorial/)
