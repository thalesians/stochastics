import unittest

import numpy as np
import statsmodels.api as sm

import thalesians.stochastics.outliers as outliers

class TestOutliers(unittest.TestCase):
    def test_outliers(self):
        random_state = np.random.RandomState(seed=42)
        nobs = 300
        sample = random_state.normal(size=nobs)
        kde = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
        kde.fit()
        self.assertFalse(outliers.isoutlier(sample, kde.bw, 1., threshold=.1, count=10000, random_state=random_state))
        self.assertTrue(outliers.isoutlier(sample, kde.bw, 100., threshold=.1, count=10000, random_state=random_state))

if __name__ == '__main__':
    unittest.main()