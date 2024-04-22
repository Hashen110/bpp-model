import pytest
from app import predict_future_days, df


def test_predict_future_days():
    # Call the function with a prediction of 3 days
    predictions = predict_future_days(3)

    # Check if the predictions match the expected values
    assert predictions == [154646.453125, 267002.34375, 399566.34375]
    # Check if the number of predictions is correct
    assert len(predictions) == 3
    for prediction in predictions:
        # Check if each prediction is a float
        assert isinstance(prediction, float)


def test_predict_future_days_zero_days():
    with pytest.raises(ValueError):
        # Call the function with 0 prediction days
        predict_future_days(0)


def test_predict_future_days_invalid_days():
    with pytest.raises(ValueError):
        # Call the function with -1 prediction days
        predict_future_days(-1)


def test_dataset():
    # Check if the dataset is loaded correctly
    assert len(df) == 2434
    # Check dataset columns
    assert list(df.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


if __name__ == "__main__":
    pytest.main()
