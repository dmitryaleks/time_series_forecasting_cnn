import unittest
import numpy as np

from forecast.timeseries_forecast_cnn import evaluate_timeseries

class TestForecast(unittest.TestCase):

    def is_close_enough(self, one, another, epsilon):
        return abs(one - another) <= epsilon

    def test_forecast(self):

        timeseries = np.arange(1000)
        prediction = evaluate_timeseries(timeseries, 20)
        print(prediction)
        self.assertTrue(self.is_close_enough(prediction, 1000, 20.0))
