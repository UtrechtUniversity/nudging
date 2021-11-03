import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from mock import patch

from nudging.correlation import equal_range, smooth_data


def test_equal_range():
    start = 1
    stop = 10
    n_step = 1
    ranges = equal_range(start, stop, n_step)
    for interval in ranges:
        result = interval
    expected = range(0, 9)
    assert result == expected

@patch('nudging.correlation.equal_range')
def test_smooth_data(interval):
    interval.return_value = [range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8, 10)]
    x = np.arange(10)
    y = np.arange(10)
    x_new, y_new = smooth_data(x, y)
    expected = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
    np.testing.assert_array_equal(x_new, expected)
    np.testing.assert_array_equal(y_new, expected)
