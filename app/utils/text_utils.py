import re
import html
import logging

logger = logging.getLogger(__name__)


def clean_review_text(text: str) -> str:
    """
    Cleans raw Amazon review text before passing to the model.

    The Kaggle dataset had a few recurring issues I noticed during EDA:
    - HTML entities like &amp; and &#39; left in review_body
    - Some reviews were copy-pasted with <br> tags still in them
    - Trailing/leading whitespace everywhere
    - Occasional runs of repeated punctuation (!!!!!!) that don't add signal

    This function handles all of that without being too aggressive —
    I deliberately kept contractions and mixed case because DistilBERT's
    tokenizer handles those fine and stripping them loses meaning.
    """
    if not isinstance(text, str):
        return ""

    # unescape HTML entities first, before stripping tags — otherwise
    # &lt;b&gt; becomes <b> which then gets stripped, which is the right order
    text = html.unescape(text)

    # strip any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # collapse repeated punctuation — "!!!!" → "!" etc.
    # kept this conservative; only collapses 3+ repeats to avoid messing with
    # things like "..." which has semantic meaning
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)

    # normalize whitespace — covers \t, \n, \r, multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Word-level truncation instead of character-level.

    Tried character truncation first but it sometimes cut mid-word which
    looked weird in the UI and could theoretically confuse the tokenizer
    on edge cases. Word-level is slightly over the char limit sometimes
    but the tokenizer will handle the rest.
    """
    words = text.split()
    if len(words) <= max_length:
        return text
    return " ".join(words[:max_length])


def validate_input(text: str, min_length: int = 10, max_length: int = 512) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message). Keeping validation logic here
    rather than in the route handlers so it's testable and reusable.
    """
    if not text or not isinstance(text, str):
        return False, "Input text is required and must be a string."

    cleaned = clean_review_text(text)

    if len(cleaned.strip()) < min_length:
        return False, f"Review text is too short (minimum {min_length} characters after cleaning)."

    # Log a warning if we're going to truncate — useful to know how often
    # users are pasting huge reviews
    word_count = len(cleaned.split())
    if word_count > max_length:
        logger.debug(f"Input has {word_count} words, will be truncated to {max_length}")

    return True, ""


def format_confidence_scores(raw_scores: list[float], labels: dict) -> list[dict]:
    """
    Converts raw softmax scores into the format the frontend expects.
    Rounds to 4 decimal places — anything more is false precision for
    a fine-tuned model at this scale.
    """
    return [
        {"label": labels[i], "score": round(float(score), 4)}
        for i, score in enumerate(raw_scores)
    ]
