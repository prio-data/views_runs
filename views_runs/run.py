
from stepshift.views import StepshiftedModels
from views_partitioning import DataPartitioner
import pandas as pd

from . import validation

class ViewsRun():
    """
    ViewsRun

    This class combines modelling and partitioning, delivering predictions in a
    familiar format and concise API.

    Takes two arguments, a views_partitioning.DataPartitioner instance, and a
    stepshift.StepshiftedModels instance. These instances determine how data is
    subset, and how modelling happens.
    """

    def __init__(self, partitioner: DataPartitioner, models: StepshiftedModels):
        self._models = models
        self._shifted_partitioner = partitioner.shift_left(self._models._steps_extent)
        self._partitioner = partitioner

    @validation.views_validate
    def fit(self, partition_name: str, timespan_name: str, data: pd.DataFrame)-> None:
        """
        Fit the model to the named partition. partition_name must be present in
        the DataPartitioner.
        """

        self._models.fit(self._shifted_partitioner(partition_name,timespan_name,data))

    @validation.views_validate
    def predict(self, partition_name: str, timespan_name: str, data: pd.DataFrame)-> pd.DataFrame:
        """
        Predict with data from the named partition. partition_name must be
        present in the DataPartitioner.
        """

        predictions = self._models.predict(
                self._shifted_partitioner(partition_name, timespan_name, data),
                combine = False)
        predictions.index.names = data.index.names
        data = data.merge(predictions, how = "left", left_index = True, right_index = True)
        return self._partitioner(partition_name,timespan_name,data)

    @property
    def models(self):
        return self._models.models
