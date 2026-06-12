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

    T5 is a seq2seq model so inference is more complex than classification —
    we need to call generate() and then decode the output token ids.
    """

    # T5 was pretrained with task prefixes. Without "summarize: " the model
    # sometimes just copies the input verbatim, which I confirmed empirically
    # during early testing. The prefix activates the right decoder behavior.
    TASK_PREFIX = "summarize: "

    def summarize(self, text: str) -> dict:
        if not model_loader.summarization_ready:
            raise RuntimeError("Summarization model is not loaded.")

        cleaned = clean_review_text(text)
        # T5's encoder can handle up to 512 tokens; truncating at word level
        # to stay safe
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

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.SUMMARIZATION_MAX_NEW_TOKENS,
                min_length=cfg.SUMMARIZATION_MIN_LENGTH,
                num_beams=cfg.SUMMARIZATION_NUM_BEAMS,
                length_penalty=cfg.SUMMARIZATION_LENGTH_PENALTY,
                # early_stopping only makes sense with beam search
                early_stopping=True,
                # no_repeat_ngram_size=3 helped prevent the model from
                # repeating phrases when I tested on longer reviews
                no_repeat_ngram_size=3,
            )

        # skip_special_tokens=True strips the </s> EOS token from the output
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "summary": summary.strip(),
            "input_length": len(truncated.split()),
            "summary_length": len(summary.split()),
        }


summarization_service = SummarizationService()
