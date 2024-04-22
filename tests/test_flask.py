import pytest
from app import create_app


@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    # other setup can go here
    yield app
    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_request_predict_without_days(client):
    response = client.get("/predict")
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    predictions = data["predictions"]
    assert len(predictions) == 30


def test_request_predict_0_days(client):
    with pytest.raises(ValueError):
        response = client.get("/predict?pred_days=0")
        assert response.status_code == 500


def test_request_404(client):
    response = client.get("/prediction")
    assert response.status_code == 404


def test_request_predict_10_days(client):
    response = client.get("/predict?pred_days=10")
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    predictions = data["predictions"]
    assert len(predictions) == 10
