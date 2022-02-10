
from typing import Callable, Tuple, Dict
import pandas as pd
from views_partitioning import data_partitioner
from stepshift import views
from views_runs import run, storage, run_result

def retrain_or_retrieve(
        store: storage.Storage,
        partitioner: data_partitioner.DataPartitioner,
        stepshifted_models: views.StepshiftedModels,
        queryset_name: str,
        partition_name: str,
        storage_name: str,
        author_name: str,
        timespan_name: str = "train",
        retrain: bool = False,
        data_preprocessing: Callable[[pd.DataFrame], pd.DataFrame] = lambda x:x) -> Tuple[run.ViewsRun, pd.DataFrame]:
    """
    retrain_or_retrieve
    ===================

    parameters:
        store (views_runs.storage.Storage)
        partitioner (views_partitioning.data_partitioner.DataPartitioner)
        stepshifted_models (stepshift.views.StepshiftedModels)
        queryset_name (str)
        partition_name (str)
        storage_name (str)
        author_name (str)
        timespan_name (str) = "train"
        retrain (bool) = False
        data_preprocessing (Callable[[pd.DataFrame], pd.DataFrame])
    returns:
        views_runs.run.ViewsRun

    This function either retrieves or trains a views run based on the passed
    specification. Also returns the fetched (and possibly processed) data.

    examples:

        # Basic usage
        #

        modelstore = Storage()
        run, data = retrain_or_retrieve(
            store              = modelstore,
            partitioner        = data_partitioner.DataPartitioner({"A":{"train":(1,100)}}),
            stepshifted_models = views.StepshiftedModels(LinearRegression(), [1,2,3,4,5,6], "depvar"),
            queryset_name      = "my-queryset",
            partition_name     = "A",
            storage_name       = "my-run",
            author_name        = "me")

        # With preprocessing
        #

        my_columns = ["a","b","c"]
        def only_abc(data: pd.DataFrame)-> pd.DataFrame:
            return data[[my_columns]]

        run, data = retrain_or_retrieve(
            store              = modelstore,
            partitioner        = data_partitioner.DataPartitioner({"A":{"train":(1,100)}}),
            stepshifted_models = views.StepshiftedModels(LinearRegression(), [1,2,3,4,5,6], "depvar"),
            queryset_name      = "my-queryset",
            partition_name     = "A",
            storage_name       = "my-run",
            author_name        = "me",
            data_preprocessing = plus_ten)
    """
    return run_result.RunResult.retrain_or_retrieve(
        store,
        partitioner,
        stepshifted_models,
        queryset_name,
        partition_name,
        storage_name,
        author_name,
        timespan_name,
        retrain,
        data_preprocessing)

def hurdle_feature_importances(data: pd.DataFrame, views_run: run.ViewsRun) -> Dict[int, pd.DataFrame]:
    """
    hurdle_feature_importances
    ==========================

    to be implemented...
    """

    return dict
