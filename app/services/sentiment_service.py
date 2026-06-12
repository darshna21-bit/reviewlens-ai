import logging
from typing import Optional

import torch
import torch.nn.functional as F

from app.models.model_loader import model_loader
from app.utils.text_utils import clean_review_text, truncate_text, format_confidence_scores
from config.settings import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class SentimentService:
    """
    Wraps DistilBERT inference. Keeping this in a service class rather than
    directly in the route handler lets me mock it cleanly in tests without
    needing an actual model file.
    """

    LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def predict(self, text: str) -> dict:
        """
        Returns the predicted sentiment label and per-class confidence scores.

        The softmax over logits gives probabilities, but I want to be clear in
        the response that these are model confidence scores, not calibrated
        probabilities — a fine-tuned classifier isn't necessarily well-calibrated
        without temperature scaling or Platt scaling.
        """
        if not model_loader.sentiment_ready:
            raise RuntimeError("Sentiment model is not loaded.")

        cleaned = clean_review_text(text)
        truncated = truncate_text(cleaned, max_length=cfg.MAX_INPUT_LENGTH)

        tokenizer = model_loader.sentiment_tokenizer
        model = model_loader.sentiment_model

        inputs = tokenizer(
            truncated,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.MAX_INPUT_LENGTH,
            padding=True,
        )

        # move inputs to the same device as the model
        inputs = {k: v.to(model_loader.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # shape: (1, num_labels) → squeeze to (num_labels,)
        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().tolist()

        predicted_class = int(torch.argmax(logits).item())
        label = self.LABELS[predicted_class]
        confidence = round(probs[predicted_class], 4)

        return {
            "label": label,
            "confidence": confidence,
            "scores": format_confidence_scores(probs, self.LABELS),
        }


# module-level instance; imported by controllers
sentiment_service = SentimentService()
