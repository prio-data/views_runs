
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

    def test_future_predict(self):
        run = ViewsRun(
                DataPartitioner({"a":{"train":(0,50),"test":(51,99)}}),
                StepshiftedModels(LinearRegression(), (1,2,3,4), outcome = "dep")
                )
        run.fit("a","train",self.mock_data)
        future = run.future_predict("a","test", self.mock_data)
        self.assertEqual(set(future.index.get_level_values(0)), {100,101,102,103})

    def test_future_point_predict(self):
        run = ViewsRun(
                DataPartitioner({"a":{"train":(0,50),"test":(51,99)}}),
                StepshiftedModels(LinearRegression(), (1,2,3,4), outcome = "dep")
                )
        run.fit("a","train",self.mock_data)
        a = run.future_point_predict(99, self.mock_data)
        self.assertEqual(set(a.index.get_level_values(0)), {100,101,102,103})

        b = run.future_point_predict(50, self.mock_data)
        self.assertEqual(set(b.index.get_level_values(0)), {51,52,53,54})
