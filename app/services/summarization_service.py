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

    def summarize(self, text: str, sentiment: str = None) -> dict:
        if not model_loader.summarization_ready:
            raise RuntimeError("Summarization model is not loaded.")

        cleaned = clean_review_text(text)
        truncated = truncate_text(cleaned, max_length=400)

        # Sentiment-Aware Prompting (Option 1) to bypass training bias
        prefix = self.TASK_PREFIX
        if sentiment == "Neutral":
            prefix = "summarize neutral review: "
        elif sentiment == "Positive":
            prefix = "summarize positive review: "
        elif sentiment == "Negative":
            prefix = "summarize negative review: "

        prefixed = prefix + truncated

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
        gen_min_length = 8
        gen_length_penalty = 1.2
        gen_num_beams = cfg.SUMMARIZATION_NUM_BEAMS

        # Option 2: Greedy decoding for neutral reviews to avoid template bias
        if sentiment == "Neutral":
            gen_num_beams = 1
            gen_min_length = 6
            gen_length_penalty = 1.0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.SUMMARIZATION_MAX_NEW_TOKENS,
                min_length=gen_min_length,
                num_beams=gen_num_beams,
                length_penalty=gen_length_penalty,
                repetition_penalty=1.2,
                early_stopping=True if gen_num_beams > 1 else False,
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
