
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from stepshift.views import StepshiftedModels 
from views_partitioning import DataPartitioner
from views_runs import ViewsRun
from unittest import TestCase

class TestRun(TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame(
                np.random.rand(2000).reshape(1000,2),
                columns = ["dep","indep"],
                index = pd.MultiIndex.from_product((range(100),range(10))))

    def test_future_predict_indices(self):
        run = ViewsRun(
                DataPartitioner({"a":{"train":(0,50),"test":(51,99)}}),
                StepshiftedModels(LinearRegression(), (1,2,3,4), outcome = "dep")
                )
        run.fit("a","train",self.mock_data)
        future = run.future_predict("a","test", self.mock_data)
        self.assertEqual(set(future.index.get_level_values(0)), {100,101,102,103})

    def test_future_point_predict_indices(self):
        run = ViewsRun(
                DataPartitioner({"a":{"train":(0,50),"test":(51,99)}}),
                StepshiftedModels(LinearRegression(), (1,2,3,4), outcome = "dep")
                )
        run.fit("a","train",self.mock_data)
        a = run.future_point_predict(99, self.mock_data)
        self.assertEqual(set(a.index.get_level_values(0)), {100,101,102,103})

        b = run.future_point_predict(50, self.mock_data)
        self.assertEqual(set(b.index.get_level_values(0)), {51,52,53,54})

    def test_future_point_predict_modelling(self):
        """
        This test checks the predict methods by expressing a simple time-trend
        """

        data = pd.DataFrame(
                np.array([
                        [0,1],
                        [1,0],
                        [0,0],
                        [0,1],
                        [0,0],
                    ]).astype(float),
                index = pd.MultiIndex.from_product([range(5), (1,)]),
                columns = ["dep","a"])

        run = ViewsRun(
                DataPartitioner({"a":{"train":(0,2)}}),
                StepshiftedModels(LinearRegression(), (1,), outcome = "dep"))

        run.fit("a","train",data)
        np.testing.assert_almost_equal(run.future_point_predict(3, data), [[1.0]])
        np.testing.assert_almost_equal(run.future_point_predict(4, data), [[0.0]])
