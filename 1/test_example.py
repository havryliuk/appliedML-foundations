from example import generate_iris_predictions


def test_simple_prediction_result():
    result = generate_iris_predictions()
    assert result.size == 150
