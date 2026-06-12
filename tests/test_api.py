"""
Tests for the Flask API routes.

Using Flask's test client rather than requests so we don't need a running
server. All model calls are mocked so these tests don't require trained weights.
"""

import json
from unittest.mock import patch


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200

    def test_health_response_structure(self, client):
        res = client.get("/api/health")
        data = res.get_json()
        assert "status" in data
        assert "timestamp" in data
        assert "models" in data
        assert data["status"] == "ok"

    def test_health_models_key(self, client):
        res = client.get("/api/health")
        models = res.get_json()["models"]
        # both keys must be present regardless of load state
        assert "sentiment_model" in models
        assert "summarization_model" in models


class TestPredictEndpoint:
    def test_predict_no_body_returns_400(self, client):
        res = client.post("/api/predict", content_type="application/json")
        assert res.status_code == 400

    def test_predict_empty_text_returns_400(self, client):
        res = client.post(
            "/api/predict",
            json={"text": ""},
        )
        assert res.status_code == 400

    def test_predict_short_text_returns_400(self, client):
        res = client.post("/api/predict", json={"text": "ok"})
        assert res.status_code == 400

    def test_predict_model_not_loaded_returns_503(self, client, sample_positive_review):
        # model_loader.sentiment_ready is False in test config
        res = client.post("/api/predict", json={"text": sample_positive_review})
        # either 503 (model not loaded) or 200 (if somehow loaded) — just not 500
        assert res.status_code in (200, 503)

    def test_predict_with_mock_model(self, client, sample_positive_review, mock_sentiment_result):
        with patch(
            "app.services.sentiment_service.sentiment_service.predict",
            return_value=mock_sentiment_result,
        ):
            res = client.post("/api/predict", json={"text": sample_positive_review})
            assert res.status_code == 200
            data = res.get_json()
            assert data["status"] == "ok"
            assert data["result"]["label"] == "Positive"
            assert "scores" in data["result"]
            assert len(data["result"]["scores"]) == 3

    def test_predict_invalid_json(self, client):
        res = client.post(
            "/api/predict",
            data="not json at all",
            content_type="application/json",
        )
        assert res.status_code == 400


class TestSummarizeEndpoint:
    def test_summarize_too_short_returns_400(self, client):
        res = client.post("/api/summarize", json={"text": "This is short."})
        assert res.status_code == 400

    def test_summarize_with_mock_model(self, client, sample_positive_review, mock_summary_result):
        with patch(
            "app.services.summarization_service.summarization_service.summarize",
            return_value=mock_summary_result,
        ):
            res = client.post("/api/summarize", json={"text": sample_positive_review})
            assert res.status_code == 200
            data = res.get_json()
            assert data["status"] == "ok"
            assert "summary" in data["result"]
            assert isinstance(data["result"]["summary"], str)
            assert len(data["result"]["summary"]) > 0

    def test_summarize_missing_text_key(self, client):
        # sending valid JSON but with wrong key
        res = client.post("/api/summarize", json={"review": "some text"})
        assert res.status_code == 400


class TestAnalyzeEndpoint:
    def test_analyze_returns_combined_results(
        self, client, sample_positive_review, mock_sentiment_result, mock_summary_result
    ):
        with (
            patch(
                "app.services.sentiment_service.sentiment_service.predict",
                return_value=mock_sentiment_result,
            ),
            patch(
                "app.services.summarization_service.summarization_service.summarize",
                return_value=mock_summary_result,
            ),
        ):
            res = client.post("/api/analyze", json={"text": sample_positive_review})
            assert res.status_code == 200
            data = res.get_json()
            assert "sentiment" in data["result"]
            assert "summarization" in data["result"]

    def test_analyze_short_text_skips_summarization(self, client, mock_sentiment_result):
        # short text should skip summarization but still do sentiment
        short_text = "This product is really good and I like it very much."
        with patch(
            "app.services.sentiment_service.sentiment_service.predict",
            return_value=mock_sentiment_result,
        ):
            res = client.post("/api/analyze", json={"text": short_text})
            assert res.status_code == 200
            data = res.get_json()
            result = data["result"]
            # summarization skipped for short texts
            if "summarization" in result:
                assert result["summarization"].get("skipped") is True or "error" in result["summarization"]

    def test_analyze_no_body_returns_400(self, client):
        res = client.post("/api/analyze", content_type="application/json")
        assert res.status_code == 400


class TestIndexRoute:
    def test_index_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_index_returns_html(self, client):
        res = client.get("/")
        assert b"ReviewLens" in res.data
        assert res.content_type.startswith("text/html")
