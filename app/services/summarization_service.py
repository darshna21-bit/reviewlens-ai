import logging
import os
import requests

import torch

from app.models.model_loader import model_loader
from app.utils.text_utils import clean_review_text, truncate_text
from config.settings import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class SummarizationService:
    """
    Wraps summarization inference. Uses a hybrid approach:
    1. Tries to call Hugging Face Serverless API with BART-large-cnn for state-of-the-art abstractive summaries.
    2. Falls back to local T5-small model if the API is offline or rate-limited.
    """

    TASK_PREFIX = "summarize: "

    def summarize(self, text: str) -> dict:
        cleaned = clean_review_text(text)
        input_word_count = len(cleaned.split())

        # 1. Try Hugging Face Serverless Inference API (BART-Large-CNN) for high-end abstractive summary
        try:
            token = os.getenv("HF_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            
            payload = {
                "inputs": cleaned,
                "parameters": {
                    "max_length": 45,
                    "min_length": 10,
                    "do_sample": False
                }
            }
            
            # Short timeout of 4 seconds so that if HF API is slow, we fall back instantly without latency
            response = requests.post(api_url, headers=headers, json=payload, timeout=4)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get("summary_text", "").strip()
                    if summary:
                        logger.info("Successfully generated summary via Hugging Face Inference API.")
                        return {
                            "summary": summary,
                            "input_length": input_word_count,
                            "summary_length": len(summary.split()),
                        }
            else:
                logger.warning(f"Inference API returned status code {response.status_code}. Falling back to local model.")
        except Exception as e:
            logger.warning(f"Inference API failed: {e}. Falling back to local model.")

        # 2. Local Fallback (Existing fine-tuned T5-small model)
        if not model_loader.summarization_ready:
            raise RuntimeError("Summarization model is not loaded.")

        # Strip introductory fluff (e.g., "I bought this last week") so the model
        # doesn't copy generic first sentences and focuses on the actual review content.
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        if len(sentences) > 1:
            first_sentence = sentences[0].lower()
            fluff_words = ["ordered", "bought", "purchased", "received", "arrived", "shipping", "delivery"]
            if any(w in first_sentence for w in fluff_words) and len(first_sentence.split()) < 10:
                cleaned = ". ".join(sentences[1:]) + "."

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

        input_word_count = len(truncated.split())

        # Refined parameters to ensure a true overall summary instead of sentence copying:
        # 1. Neutralize length penalty (1.0) so the model doesn't bloat text to satisfy a length quota.
        # 2. Set min_length to 6 for short/medium reviews, and 10 for longer reviews to prevent one-word summaries.
        if input_word_count < 50:
            gen_min_length = 6
            gen_length_penalty = 1.0
        else:
            gen_min_length = 10
            gen_length_penalty = 1.0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.SUMMARIZATION_MAX_NEW_TOKENS,
                min_length=gen_min_length,
                num_beams=cfg.SUMMARIZATION_NUM_BEAMS,
                length_penalty=gen_length_penalty,
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
