
# views-runs 

This package is meant to help views researchers with training models, by
providing a common interface for data partitioning and stepshift model
training. It also functions as a central hub package for other classes and
functions used by views researchers, 
including [stepshift](https://github.com/prio-data/stepshift) (StepshiftedModels)
and [views_partitioning](https://github.com/prio-data/views_partitioning) (DataPartitioner).

## Installation

To install `views-runs`, use pip:

```
pip install views-runs
```

This also installs the vendored libraries `stepshift` and `views_partitioning`.

## Usage

There are notebooks that show various workflows with `views_runs` and the
vendored libraries:

* [BasicExample.ipynb](examples/BasicExample.ipynb)

See respective repositories for further documentation vendored libraries:

* [stepshift](https://github.com/prio-data/stepshift)
* [views_partitioning](https://github.com/prio-data/views_partitioning).
