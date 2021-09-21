import pandas as pd
from pandas._testing import assert_frame_equal
from src.propensity_score import get_pscore, check_weights


def test_get_pscore():
	# initialise data of lists.
	data = {
		'gender':[0, 0, 1, 1],
		'age':[30, 30, 30, 30],
		'outcome': [1, 2, 3, 4],
		'nudge': [0, 1, 0, 1]
	}

	# Create DataFrame
	df = pd.DataFrame(data)
	result = get_pscore(df)
	print(result)
	expected = df.copy(deep=True)
	expected['pscore'] = [0.237304, 0.457155, 0.626766, 0.678775]
	print(expected)
	assert_frame_equal(result, expected)



# def test_check_weights():

#     assert result == expected