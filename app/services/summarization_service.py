import logging

import torch

from app.models.model_loader import model_loader
from app.utils.text_utils import clean_review_text, truncate_text
from config.settings import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class SummarizationService:
    """
    Wraps T5-small inference for abstractive summarization.
    Uses the locally fine-tuned custom T5-small model.
    """

    TASK_PREFIX = "summarize: "

    def summarize(self, text: str) -> dict:
        if not model_loader.summarization_ready:
            raise RuntimeError("Summarization model is not loaded.")

        cleaned = clean_review_text(text)
        truncated = truncate_text(cleaned, max_length=400)

        prefixed = self.TASK_PREFIX + truncated

        tokenizer = model_loader.summarization_tokenizer
        model = model_loader.summarization_model

        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(model_loader.device) for k, v in inputs.items()}

        input_word_count = len(truncated.split())

        # Clean, optimized generation parameters for your fine-tuned model:
        # We set min_length to 6 to prevent one-word summaries, and length_penalty to 1.0
        # to ensure the model doesn't copy sentences to meet a long length quota.
        gen_min_length = 6
        gen_length_penalty = 1.0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.SUMMARIZATION_MAX_NEW_TOKENS,
                min_length=gen_min_length,
                num_beams=cfg.SUMMARIZATION_NUM_BEAMS,
                length_penalty=gen_length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        # skip_special_tokens=True strips the </s> EOS token from the output
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "summary": summary.strip(),
            "input_length": input_word_count,
            "summary_length": len(summary.split()),
        }


summarization_service = SummarizationService()
