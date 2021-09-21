import pandas as pd
from pandas._testing import assert_frame_equal
from src.propensity_score import get_pscore, check_weights


def test_get_pscore():
	data = {
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1]
	}

	# Create DataFrame
	df = pd.DataFrame(data)
	result = get_pscore(df)
	expected = df.copy(deep=True)
	expected['pscore'] = [0.5, 0.5, 0.5, 0.5]
	assert_frame_equal(result, expected)

def test_check_weights():
	data = {
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1],
		'pscore': [0.5, 0.5, 0.5, 0.5]
	}

	# Create DataFrame
	df = pd.DataFrame(data)
	result = check_weights(df)
	expected = (4, 4, 4)
	assert result == expected