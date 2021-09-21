import pandas as pd
from pandas._testing import assert_frame_equal
from nudging.propensity_score import get_pscore, check_weights, get_ate, get_psw_ate


# Create DataFrame
data_frame = pd.DataFrame({
	'gender':[0, 0, 1, 1],
	'age':[30, 30, 30, 30],
	'outcome': [1, 2, 3, 4],
	'nudge': [0, 1, 0, 1]
})

def test_get_pscore():
	result = get_pscore(data_frame)
	expected = data_frame.copy(deep=True)
	expected['pscore'] = [0.5, 0.5, 0.5, 0.5]
	assert_frame_equal(result, expected)

def test_check_weights():
	df = data_frame.copy(deep=True)
	df['pscore'] = [0.5, 0.5, 0.5, 0.5]
	result = check_weights(df)
	expected = (4, 4, 4)
	assert result == expected

def test_get_ate():
	result = get_ate(data_frame)
	expected = 1.0
	assert result == expected

def test_get_psw_ate():
	df = data_frame.copy(deep=True)
	df['pscore'] = [0.5, 0.5, 0.5, 0.5]
	result = get_psw_ate(df)
	expected = 1.0
	assert result == expected