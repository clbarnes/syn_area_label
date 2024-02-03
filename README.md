# syn_area_label

For the latest version, see here: <https://github.com/clbarnes/syn_area_label>

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

Then install this package:

```sh
pip install git+https://github.com/clbarnes/syn_area_label.git
```

If this is a dependency in another project, choose the revision (tag, commit, or branch)
and use the line `syn_area_label @ git+https://github.com/clbarnes/syn_area_label.git@REVISION_GOES_HERE`
in your requirements file or project metadata.

The workflow is this:

1. Identify the synaptic edges (presynapse -> connector -> postsynapse) you are interested in
2. Build an `syn_area_label.EdgeTables` object from them, querying CATMAID
3. Open your image volume as an `xarray.DataArray` (pymaid has a utility for opening certain CATMAID stacks this way)
4. Pass these to `syn_area_label.cache_rois`

This will generate one CREMI-format HDF5 file per synaptic edge, and some extra tables.
This should be annotated according to [`painting_instructions.md`](./painting_instructions.md).

```python
#!/usr/bin/env python3
from pathlib import Path

from syn_area_label import EdgeTables, cache_rois
from pymaid.stack import Stack

skids = [
    16015738,
    16627773,
    16629931,
    14522124,
]
creds = json.loads(Path("path/to/credentials.json").read_text())
cm = pymaid.CatmaidInstance()

stack = Stack.from_catmaid("L1 CNS 0-tilt")
```
