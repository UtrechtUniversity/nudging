import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from nudging.model.propensity_score import get_pscore, check_weights, get_ate, get_psw_ate, \
	perform_matching, obtain_match_details, match_ps

# Create DataFrame


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
	result = get_pscore(df)

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

def test_perform_matching():
	"""Check no match for treatment group"""
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		'age':[39, 40, 50, 55, 39, 40, 50, 55, 39, 40, 50, 55],
		'outcome': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
		'nudge': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	}).reset_index()
	indexes = np.array(
		[[0, 1, 2, 3],
		[0, 1, 2, 3],
		[2, 1, 0, 3],
		[2, 1, 0, 3]]
	)
	result = perform_matching(df.loc[1], indexes, df)
	expected = 0
	assert result == expected

def test_perform_matching_nan():
	"""Check no match for control group"""
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		'age':[39, 40, 50, 55, 39, 40, 50, 55, 39, 40, 50, 55],
		'outcome': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
		'nudge': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	}).reset_index()
	indexes = np.array(
		[[0, 1, 2, 3],
		[0, 1, 2, 3],
		[2, 1, 0, 3],
		[2, 1, 0, 3]]
	)
	result = perform_matching(df.loc[0], indexes, df)
	assert np.isnan(result)

def test_obtain_match_details():
	df = pd.DataFrame({
		'gender':[0, 0, 1, 1],
		'age':[39, 40, 50, 55],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1],
		'matched_element': [np.nan, 0, np.nan, 1]
	})
	result = obtain_match_details(df.loc[1], df, "age")
	expected = 39
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
	result = match_ps(df).reset_index().drop(columns=['index'])
	expected = pd.DataFrame({
		'gender':[0, 1, 0, 1, 0, 1],
		'age':[40, 55, 40, 55, 40, 55],
		'outcome': [2, 4, 2, 4, 2, 4],
		'control': [1, 3, 1, 3, 1, 3]
	})
	assert_frame_equal(result, expected)
