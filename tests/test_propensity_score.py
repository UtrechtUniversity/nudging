import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from nudging.model.propensity_score import get_pscore, check_weights, get_ate, get_psw_ate, match_ps


def test_get_pscore_same_age():
	data_frame = pd.DataFrame({
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1]
	})
	result = get_pscore(data_frame)
	expected = data_frame.copy(deep=True)
	expected['pscore'] = [0.5, 0.5, 0.5, 0.5]
	assert_frame_equal(result, expected)

def test_get_pscore_different_age():
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		'age':[39, 40, 50, 55, 39, 40, 50, 55, 39, 40, 50, 55],
		'outcome': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
		'nudge': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	})
	result = get_pscore(df, solver='lbfgs')

	expected = pd.DataFrame({
		'gender':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		'age':[39, 40, 50, 55, 39, 40, 50, 55, 39, 40, 50, 55],
		'outcome': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
		'nudge': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		'pscore': [0.385941, 0.412338, 0.535314, 0.666406, 0.385941, 0.412338, 0.535314, 0.666406, 0.385941, 0.412338, 0.535314, 0.666406]
	})

	assert_frame_equal(result, expected)

def test_check_weights():
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1],
		'pscore': [0.5, 0.5, 0.5, 0.5]
	})
	result = check_weights(df)
	expected = (4, 4, 4)
	assert result == expected

def test_get_ate():
	data_frame = pd.DataFrame({
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1]
	})
	result = get_ate(data_frame)
	expected = 1.0
	assert result == expected

def test_get_psw_ate():
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1],
		'pscore': [0.5, 0.5, 0.5, 0.5]
	})
	result = get_psw_ate(df)
	expected = 1.0
	assert result == expected

def test_match_ps():
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		'age':[39, 40, 50, 55, 39, 40, 50, 55, 39, 40, 50, 55],
		'outcome': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
		'nudge': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
		'pscore': [0.385941, 0.412338, 0.535314, 0.666406, 0.385941, 0.412338, 0.535314, 0.666406, 0.385941, 0.412338, 0.535314, 0.666406
]
	})
	result = match_ps(df)
	expected = pd.DataFrame({
		'gender':[0, 1, 0, 1, 0, 1],
		'age':[40, 55, 40, 55, 40, 55],
		'outcome': [2, 4, 2, 4, 2, 4],
		'control': [1, 3, 1, 3, 1, 3]
	})
	assert_frame_equal(result, expected)
