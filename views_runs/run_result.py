import pandas as pd
from views_schema import ModelMetadata
from stepshift import views
from views_partitioning import data_partitioner
from views_runs import storage, run

class RunResult():
    """
    RunResult
    =========

    A run result organizes the various components that constitute a views run.
    The components are:

    * ViewsRun (.run), which combines Stepshift model estimation and data
      subsetting via the DataPartitioner,
    * ModelMetadata (.metadata), which contains a description of both the
      model, data and context of the run

    These components can be described in diagram form like this:

        ┌─────────┐
        │         │
        │RunResult│─────────────────────────┐
        │         │───────────┐.metadata    │
        └─────────┘           ▼             │
             │         ┌─────────────┐      │
             │         │ModelMetadata│      │
             │.run     └─────────────┘      │.data
             ▼                              ▼
        ┌────────┐                     ┌───────┐
        │ViewsRun│◄────────────────────│Dataset│
        └────────┘                     └───────┘
             │
             │
             │
             │.partitioner
             ▼
        ┌───────────────┐
        │DataPartitioner│
        └───────────────┘

    A RunResult should be instantiated via the RunResult.retrain_or_retrieve
    classmethod. See its docstring for how to do so.

    parameters:
        views_run (views_runs.run.ViewsRun)
        metadata (views_schema.ModelMetadata)
        preprocessing (Callable[[pd.DataFrame], pd.DataFrame])

    attributes:
        retrained (bool): Whether the ViewsRun was retrained, overwriting an existing run
        data (pd.DataFrame): Dataframe corresponding to queryset defined in the metadata
        run (views_run.run.ViewsRun)
        metadata (views_schema.ModelMetadata)
    """

    def __init__(self, views_run: run.ViewsRun, metadata: ModelMetadata, data: pd.DataFrame):
        self.run            = views_run
        self.metadata       = metadata
        self.data          = data
        self.retrained      = False

    @classmethod
    def retrain_or_retrieve(
            cls:                "RunResult",
            store:              storage.Storage,
            partitioner:        data_partitioner.DataPartitioner,
            stepshifted_models: views.StepshiftedModels,
            dataset:            pd.DataFrame,
            queryset_name:      str,
            partition_name:     str,
            storage_name:       str,
            author_name:        str,
            timespan_name:      str = "train",
            retrain:            bool = False
            ) -> "RunResult":
        """
        retrain_or_retrieve
        ===================

        parameters:
            store              (views_runs.storage.Storage)
            partitioner        (views_partitioning.data_partitioner.DataPartitioner)
            stepshifted_models (stepshift.views.StepshiftedModels)
            dataset            (pandas.DataFrame)
            queryset_name      (str)
            partition_name     (str)
            storage_name       (str)
            author_name        (str)
            timespan_name      (str) = "train"
            retrain            (bool) = False
        returns:
            views_runs.run.ViewsRun

        Create a new RunResult by either training a views run, or retrieving it
        from storage if a run corresponding to the storage_name is found.

        examples:

            # Basic usage
            #
            from viewser.operations import fetch
            from views_runs import run_result, storage

            modelstore = storage.Storage()
            dataset = fetch("my-queryset")

            result = run_result.RunResult.retrain_or_retrieve(
                store              = modelstore,
                partitioner        = data_partitioner.DataPartitioner({"A":{"train":(1,100)}}),
                stepshifted_models = views.StepshiftedModels(LinearRegression(), [1,2,3,4,5,6], "depvar"),
                dataset            = dataset
                queryset_name      = "my-queryset",
                partition_name     = "A",
                storage_name       = "my-run",
                author_name        = "me")
        """

        views_run = run.ViewsRun(partitioner, stepshifted_models)

        metadata = views_run.create_model_metadata(
                author = author_name,
                queryset_name = queryset_name,
                training_partition_name = partition_name,
                training_timespan_name = timespan_name)

        result = cls(views_run, metadata, dataset)

        exists = store.exists_with_metadata(storage_name, metadata)
        if retrain or not exists:
            result.retrained = exists
            result._fit(partition_name, timespan_name)
            store.store(storage_name, result.run, result.metadata, overwrite = True)
        else:
            result.run = store.retrieve(storage_name)

        return result

    def _fit(self, partition_name, timespan_name):
        """
        _fit
        ====

        Fits the associated ViewsRun object using partitioning and data defined
        by the metadata instance.

        """
        self.run.fit(partition_name, timespan_name, self.data)

    def __repr__(self):
        return f"RunResult(training_date = {self.metadata.training_date.strftime('%Y-%m-%d')})"
