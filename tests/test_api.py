from fastapi.testclient import TestClient

from app.main import _recommendation, _risk_level, app

client = TestClient(app)


def test_risk_levels():
    assert _risk_level(0.9) == "ALTO"
    assert _risk_level(0.45) == "MEDIO"
    assert _risk_level(0.1) == "BAJO"


def test_recommendation_covers_all_levels():
    for level in ("ALTO", "MEDIO", "BAJO"):
        assert isinstance(_recommendation(level), str)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


def test_root_endpoint():
    assert client.get("/").status_code == 200
