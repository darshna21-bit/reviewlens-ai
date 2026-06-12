"""
Standalone evaluation for the T5-small summarization model.
Computes ROUGE scores and saves sample outputs.

Run: python training/summarization/evaluate_summarization.py
"""

import logging
import sys
from pathlib import Path

import evaluate
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.utils.text_utils import clean_review_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("saved_models/summarization")
TEST_DATA_PATH = Path("data/processed/test.csv")
RESULTS_DIR = Path("training/summarization/results")
TASK_PREFIX = "summarize: "


def run_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run train_summarization.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading model and tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained(str(MODEL_PATH))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()

    logger.info("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["review_body"] = test_df["review_body"].apply(clean_review_text)

    if "review_headline" not in test_df.columns:
        logger.error("review_headline column not found. Cannot evaluate without reference summaries.")
        sys.exit(1)

    test_df = test_df.dropna(subset=["review_body", "review_headline"])
    test_df = test_df[test_df["review_headline"].str.strip().str.len() > 5]
    test_df = test_df[test_df["review_body"].str.split().str.len() > 15]

    logger.info(f"Evaluating on {len(test_df)} samples...")

    predictions = []
    references = []
    batch_size = 16

    for start in range(0, len(test_df), batch_size):
        batch = test_df.iloc[start:start + batch_size]
        texts = [TASK_PREFIX + t for t in batch["review_body"].tolist()]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])
        references.extend(batch["review_headline"].tolist())

        if (start // batch_size) % 5 == 0:
            logger.info(f"  {min(start + batch_size, len(test_df))}/{len(test_df)} processed")

    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    logger.info("\n" + "=" * 50)
    logger.info("ROUGE EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    metrics_path = RESULTS_DIR / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    logger.info(f"\nMetrics saved to {metrics_path}")

    samples_path = RESULTS_DIR / "sample_summaries.txt"
    with open(samples_path, "w") as f:
        for i in range(min(20, len(predictions))):
            f.write(f"Sample {i+1}\n")
            f.write(f"Input (truncated): {test_df['review_body'].iloc[i][:200]}...\n")
            f.write(f"Reference:         {references[i]}\n")
            f.write(f"Generated:         {predictions[i]}\n")
            f.write("-" * 60 + "\n")
    logger.info(f"Sample summaries saved to {samples_path}")


if __name__ == "__main__":
    run_evaluation()
