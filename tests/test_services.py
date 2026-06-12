"""
Unit tests for SentimentService and SummarizationService.
Model calls are mocked — these tests verify the service logic,
not the model weights.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.sentiment_service import SentimentService
from app.services.summarization_service import SummarizationService
from app.utils.text_utils import clean_review_text, validate_input, truncate_text


# ── Text utility tests ────────────────────────────────────────────────────────

class TestTextUtils:
    def test_clean_removes_html_tags(self):
        dirty = "This is <b>great</b> product!"
        assert "<b>" not in clean_review_text(dirty)

    def test_clean_unescapes_html_entities(self):
        escaped = "I &amp; my family loved it"
        result = clean_review_text(escaped)
        assert "&amp;" not in result
        assert "&" in result

    def test_clean_collapses_whitespace(self):
        messy = "great   product\t\treally"
        result = clean_review_text(messy)
        assert "  " not in result

    def test_clean_handles_empty_string(self):
        assert clean_review_text("") == ""

    def test_clean_handles_non_string(self):
        assert clean_review_text(None) == ""
        assert clean_review_text(123) == ""

    def test_validate_rejects_empty(self):
        valid, msg = validate_input("")
        assert not valid
        assert msg != ""

    def test_validate_rejects_too_short(self):
        valid, msg = validate_input("hi")
        assert not valid

    def test_validate_accepts_normal_review(self):
        review = "This product worked exactly as described and arrived on time."
        valid, msg = validate_input(review)
        assert valid
        assert msg == ""

    def test_truncate_at_word_boundary(self):
        words = " ".join([f"word{i}" for i in range(600)])
        result = truncate_text(words, max_length=512)
        assert len(result.split()) <= 512

    def test_truncate_does_not_cut_short_text(self):
        short = "only a few words here"
        assert truncate_text(short, max_length=512) == short


# ── SentimentService tests ────────────────────────────────────────────────────

class TestSentimentService:
    def _make_mock_loader(self, sentiment_ready=True):
        """Helper to build a mock model_loader with predictable outputs."""
        mock_loader = MagicMock()
        mock_loader.sentiment_ready = sentiment_ready
        mock_loader.device = torch.device("cpu")

        # mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3474, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        }
        mock_loader.sentiment_tokenizer = mock_tokenizer

        # mock model output — logits shaped (1, 3)
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # positive class wins: [0.1, 0.15, 0.75]
        mock_outputs.logits = torch.tensor([[0.5, 0.8, 3.2]])
        mock_model.return_value = mock_outputs
        mock_loader.sentiment_model = mock_model

        return mock_loader

    def test_predict_raises_when_model_not_loaded(self):
        service = SentimentService()
        with patch("app.services.sentiment_service.model_loader") as mock_loader:
            mock_loader.sentiment_ready = False
            with pytest.raises(RuntimeError, match="not loaded"):
                service.predict("This product is amazing and I love it very much.")

    def test_predict_returns_expected_keys(self):
        service = SentimentService()
        mock_loader = self._make_mock_loader()

        with patch("app.services.sentiment_service.model_loader", mock_loader):
            result = service.predict(
                "This product is amazing, the build quality is excellent and works perfectly."
            )

        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert result["label"] in ("Negative", "Neutral", "Positive")
        assert 0 <= result["confidence"] <= 1
        assert len(result["scores"]) == 3

    def test_predict_scores_sum_to_one(self):
        service = SentimentService()
        mock_loader = self._make_mock_loader()

        with patch("app.services.sentiment_service.model_loader", mock_loader):
            result = service.predict(
                "I purchased this last month and it has been working flawlessly ever since."
            )

        total = sum(s["score"] for s in result["scores"])
        assert abs(total - 1.0) < 0.01  # softmax should sum to 1

    def test_predict_correct_label_from_logits(self):
        # logits [0.5, 0.8, 3.2] → argmax is index 2 → "Positive"
        service = SentimentService()
        mock_loader = self._make_mock_loader()

        with patch("app.services.sentiment_service.model_loader", mock_loader):
            result = service.predict("This product is absolutely wonderful and exceeded all expectations.")

        assert result["label"] == "Positive"


# ── SummarizationService tests ────────────────────────────────────────────────

class TestSummarizationService:
    def _make_mock_loader(self, summarization_ready=True):
        mock_loader = MagicMock()
        mock_loader.summarization_ready = summarization_ready
        mock_loader.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        # decode returns the expected summary string
        mock_tokenizer.decode = MagicMock(return_value="Good product with solid build quality.")
        mock_loader.summarization_tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
        mock_loader.summarization_model = mock_model

        return mock_loader

    def test_summarize_raises_when_model_not_loaded(self):
        service = SummarizationService()
        with patch("app.services.summarization_service.model_loader") as mock_loader:
            mock_loader.summarization_ready = False
            with pytest.raises(RuntimeError, match="not loaded"):
                service.summarize(
                    "This is a longer review about a product that I used for several weeks "
                    "before writing this review. Overall it performed well in most scenarios."
                )

    def test_summarize_returns_expected_keys(self):
        service = SummarizationService()
        mock_loader = self._make_mock_loader()

        with patch("app.services.summarization_service.model_loader", mock_loader):
            result = service.summarize(
                "I have been using this product for about three weeks now and I am very happy "
                "with the results. The build quality is solid and it performs as expected. "
                "Would definitely recommend to others who are looking for something reliable."
            )

        assert "summary" in result
        assert "input_length" in result
        assert "summary_length" in result
        assert isinstance(result["summary"], str)
        assert result["input_length"] > 0

    def test_summarize_prepends_task_prefix(self):
        """The tokenizer should receive input with 'summarize: ' prefix."""
        service = SummarizationService()
        mock_loader = self._make_mock_loader()

        with patch("app.services.summarization_service.model_loader", mock_loader):
            service.summarize(
                "The product arrived in good condition and works as expected. "
                "Nothing spectacular but it gets the job done every time I use it."
            )

        # first arg to tokenizer should contain the task prefix
        call_args = mock_loader.summarization_tokenizer.call_args
        input_text = call_args[0][0]  # positional first arg
        assert "summarize:" in input_text.lower()
