
from unittest import TestCase
import pandas as pd
import numpy as np
from views_runs.utilities import resample

class TestViews2Utilities(TestCase):
    def test_resample(self):
        data = pd.DataFrame(np.random.choice([0,1],100000,p = [.9,.1]), columns = ["val"])
        resampled = resample(data, ["val"], 1, .5)
        mean_diff = abs(resampled.val.mean() - (data.val.mean()*2))
        self.assertLess(mean_diff, .02)
