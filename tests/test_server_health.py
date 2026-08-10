from __future__ import annotations

import llm_client


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise llm_client.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


def test_mlx_server_up_requires_configured_model(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response({"data": [{"id": llm_client.MLX_MODEL}]})

    monkeypatch.setattr(llm_client.requests, "get", fake_get)

    assert llm_client.mlx_server_up() is True
    assert calls[0][0] == f"{llm_client.MLX_BASE_URL.rstrip('/')}/models"
    assert calls[0][1]["timeout"] == 0.75


def test_mlx_server_up_rejects_wrong_service_or_model(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests,
        "get",
        lambda *args, **kwargs: _Response({"data": [{"id": "some-other-model"}]}),
    )
    assert llm_client.mlx_server_up() is False

    monkeypatch.setattr(
        llm_client.requests,
        "get",
        lambda *args, **kwargs: _Response({"not": "an OpenAI models response"}),
    )
    assert llm_client.mlx_server_up() is False
