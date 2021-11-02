import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from nudging.correlation import equal_range


def test_equal_range():
    start = 1
    stop = 10
    n_step = 1
    ranges = equal_range(start, stop, n_step)
    for interval in ranges:
        result = interval
    expected = range(0, 9)
    assert result == expected