"""FastAPI smoke test.

Production code is always real-model — there is no mock-mode flag.
Tests use the ``fake_pipeline`` fixture to inject a Pipeline with
lightweight in-memory doubles for every model slot, so the test suite
stays fast and offline. See ``tests/_fakes.py``.
"""
def test_api_health_and_query(fake_pipeline, monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api_app_module
    monkeypatch.setattr(api_app_module, "Pipeline", lambda config=None: fake_pipeline)

    app = api_app_module.create_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "index_loaded" in body

        r = client.post("/index", json={"documents": ["Paris is the capital of France."]})
        assert r.status_code == 200, r.text
        r = client.post("/query", json={"query": "What is the capital of France?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["query"]
        assert body["decision"] in {"PASS", "REGEN", "ABSTAIN"}
        assert isinstance(body.get("claims"), list)
        assert isinstance(body.get("evidence"), list)


def test_api_metrics_endpoint(fake_pipeline, monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api_app_module
    monkeypatch.setattr(api_app_module, "Pipeline", lambda config=None: fake_pipeline)

    app = api_app_module.create_app()
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "rows" in r.json()


def test_api_structure_endpoint(fake_pipeline, monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api_app_module
    monkeypatch.setattr(api_app_module, "Pipeline", lambda config=None: fake_pipeline)

    app = api_app_module.create_app()
    with TestClient(app) as client:
        r = client.get("/structure?depth=3")
        assert r.status_code == 200
        body = r.json()
        assert body["root"]
        assert body["total_files"] >= 1
        names = {c["name"] for c in body["tree"]["children"]}
        assert "src" in names
        assert ".venv" not in names