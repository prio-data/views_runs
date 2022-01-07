
from unittest import TestCase
import pandas as pd
import numpy as np
from views_runs.stats import resample, resample_df

class TestStats(TestCase):
    def test_resample(self):
        x = np.random.choice([True, False], 100000, p = [.5, .5])
        resampled = x[resample(x, .4)]
        self.assertAlmostEqual(resampled.mean(), .4, 2)

    """
    def test_df_resample(self):
        x = pd.DataFrame(np.random.choice([1,0], 100000, p = [.1, .9]), columns = ["val"])
        self.assertAlmostEqual(resample_df(x, .2, "val").val.mean(),  .2, 2)
    """
