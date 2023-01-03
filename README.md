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

The library offers a class imported at `views_runs.ViewsRun`, that wraps the to
central components of a ViEWS 3 run: A partitioning scheme expressed via a
`views_partitioning.DataPartitioner` instance, and a stepshifted modelling
process expressed via a `stepshift.views.StepshiftedModels` instance. 
For documentation on the data partitioner, see 
[views_partitioning](https://www.github.com/prio-data/views_partitioning). For documentation on stepshifted modelling, see 
[views.StepshiftedModels](https://github.com/prio-data/viewser/wiki/Stepshift).


The wrapper takes care of applying these two classes to your data, in order to
produce predictions in a familiar and predictable format, as well as ensuring
that there is no overlap between training and testing partitions.
Instantiating a run requires instances of both of these classes, like so:

```
run = ViewsRun(
   DataPartitioner({"A":{"train":(1,100),"test":(101,200)}}),
   StepshiftedModels(LogisticRegression,[1,2,3,4,5,6],"my_dependent_variable"),
)
```

This instance can then be applied to a [time-unit indexed
dataframe](https://github.com/prio-data/viewser/wiki/DataConventions#time-unit-indexed-pandas-dataframes)
to train the models, and produce predictions for the timespans defined in the data partitioner:

```
run.fit("A","train",dataframe)
predictions = run.predict("A","test",dataframe)
```

## Examples

There are notebooks that show various workflows with `views_runs` and the
vendored libraries:

* [BasicExample.ipynb](examples/BasicExample.ipynb)

## Funding

The contents of this repository is the outcome of projects that have received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 694640, *ViEWS*) and Horizon Europe (Grant agreement No. 101055176, *ANTICIPATE*; and No. 101069312, *ViEWS* (ERC-2022-POC1)), Riksbankens Jubileumsfond (Grant agreement No. M21-0002, *Societies at Risk*), Uppsala University, Peace Research Institute Oslo, the United Nations Economic and Social Commission for Western Asia (*ViEWS-ESCWA*), the United Kingdom Foreign, Commonwealth & Development Office (GSRA – *Forecasting Fatalities in Armed Conflict*), the Swedish Research Council (*DEMSCORE*), the Swedish Foundation for Strategic Environmental Research (*MISTRA Geopolitics*), the Norwegian MFA (*Conflict Trends* QZA-18/0227), and the United Nations High Commissioner for Refugees (*the Sahel Predictive Analytics project*).
