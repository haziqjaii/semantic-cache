"""
Tests for the playground page and the endpoints it relies on.
"""

import re

from fastapi.testclient import TestClient

from semcache.main import app


def test_root_redirects_to_playground():
    resp = TestClient(app).get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/playground"


def test_playground_is_served_as_html():
    resp = TestClient(app).get("/playground")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<title>Semantic Cache Playground</title>" in resp.text


def test_playground_calls_endpoints_that_exist():
    """Every API path the page fetches must be a real route."""
    page = TestClient(app).get("/playground").text
    fetched = set(re.findall(r'fetch\("(/[^"?]+)', page))
    routes = set(app.openapi()["paths"])

    assert fetched == {"/v1/chat/completions", "/v1/analytics", "/v1/cache/stats",
                       "/v1/cache/invalidate", "/v1/cache/entries"}
    assert fetched <= routes


def test_playground_never_injects_html():
    """Cached prompts and answers are user text: the page must only set textContent."""
    page = TestClient(app).get("/playground").text
    assert "innerHTML" not in page
    assert "insertAdjacentHTML" not in page
    assert "document.write" not in page
