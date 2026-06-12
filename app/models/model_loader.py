import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton-ish model registry. The pattern here is: load everything once
    at app startup, hold references, serve them to services on demand.

    I initially had the services load their own models but that caused ~3s
    latency on the first request to each endpoint. Moving to startup loading
    makes the first request fast and keeps RAM usage predictable.

    On Render's free tier (512MB RAM) this is tight with both models loaded.
    If memory becomes an issue, the summarization model can be lazy-loaded
    since it's less commonly hit than sentiment.
    """

    def __init__(self):
        self._sentiment_model = None
        self._sentiment_tokenizer = None
        self._summarization_model = None
        self._summarization_tokenizer = None
        self._device = None
        self._sentiment_loaded = False
        self._summarization_loaded = False

    @property
    def device(self):
        if self._device is None:
            # MPS check for Apple Silicon — useful during local dev on M1/M2
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
            logger.info(f"Using device: {self._device}")
        return self._device

    def load_sentiment_model(self, model_path: str) -> bool:
        """
        Loads DistilBERT from a local checkpoint directory.
        Returns True on success so the app can flag which models are available.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning(
                f"Sentiment model path {model_path} doesn't exist. "
                "Sentiment endpoints will return 503 until the model is trained."
            )
            return False

        try:
            logger.info(f"Loading sentiment model from {model_path}...")
            self._sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self._sentiment_model = DistilBertForSequenceClassification.from_pretrained(
                model_path
            )
            self._sentiment_model.to(self.device)
            self._sentiment_model.eval()  # disables dropout — matters for consistent inference
            self._sentiment_loaded = True
            logger.info("Sentiment model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            return False

    def load_summarization_model(self, model_path: str) -> bool:
        path = Path(model_path)
        if not path.exists():
            logger.warning(
                f"Summarization model path {model_path} doesn't exist. "
                "Summarization endpoints will return 503 until the model is trained."
            )
            return False

        try:
            logger.info(f"Loading summarization model from {model_path}...")
            # T5TokenizerFast needs sentencepiece — make sure it's installed
            self._summarization_tokenizer = T5TokenizerFast.from_pretrained(model_path)
            self._summarization_model = T5ForConditionalGeneration.from_pretrained(
                model_path
            )
            self._summarization_model.to(self.device)
            self._summarization_model.eval()
            self._summarization_loaded = True
            logger.info("Summarization model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            return False

    @property
    def sentiment_model(self) -> Optional[DistilBertForSequenceClassification]:
        return self._sentiment_model

    @property
    def sentiment_tokenizer(self) -> Optional[DistilBertTokenizerFast]:
        return self._sentiment_tokenizer

    @property
    def summarization_model(self) -> Optional[T5ForConditionalGeneration]:
        return self._summarization_model

    @property
    def summarization_tokenizer(self) -> Optional[T5TokenizerFast]:
        return self._summarization_tokenizer

    @property
    def sentiment_ready(self) -> bool:
        return self._sentiment_loaded

    @property
    def summarization_ready(self) -> bool:
        return self._summarization_loaded

    def get_status(self) -> dict:
        return {
            "sentiment_model": "loaded" if self._sentiment_loaded else "not_loaded",
            "summarization_model": "loaded" if self._summarization_loaded else "not_loaded",
            "device": str(self.device),
        }


# Module-level singleton — imported by services
model_loader = ModelLoader()
