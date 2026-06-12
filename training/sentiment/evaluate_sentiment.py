"""
Standalone evaluation script for the sentiment model.
Run this after training to regenerate metrics and the confusion matrix.

Run: python training/sentiment/evaluate_sentiment.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.utils.text_utils import clean_review_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("saved_models/sentiment")
TEST_DATA_PATH = Path("data/processed/test.csv")
RESULTS_DIR = Path("training/sentiment/results")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def run_evaluation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run train_sentiment.py first.")
        sys.exit(1)

    if not TEST_DATA_PATH.exists():
        logger.error(f"Test data not found at {TEST_DATA_PATH}. Run data_pipeline.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()

    logger.info("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["review_body"] = test_df["review_body"].apply(clean_review_text)
    test_df = test_df.dropna(subset=["review_body", "label"])

    logger.info(f"Running inference on {len(test_df)} samples...")
    all_preds = []
    batch_size = 32

    for start in range(0, len(test_df), batch_size):
        batch_texts = test_df["review_body"].iloc[start:start + batch_size].tolist()

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(start + batch_size, len(test_df))}/{len(test_df)}")

    true_labels = test_df["label"].tolist()

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        true_labels, all_preds, average="macro", zero_division=0
    )

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:         {acc:.4f}")
    logger.info(f"Precision (macro): {p:.4f}")
    logger.info(f"Recall (macro):    {r:.4f}")
    logger.info(f"F1 (macro):        {f1:.4f}")
    logger.info("\nPer-class report:")
    logger.info(classification_report(true_labels, all_preds, target_names=LABEL_NAMES))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(true_labels, all_preds)
    # normalize so the colours reflect proportions, not raw counts
    # (otherwise the dominant positive class drowns out the others visually)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DistilBERT Sentiment Classifier — Confusion Matrix", fontsize=14)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax1
    )
    ax1.set_title("Raw Counts")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax2
    )
    ax2.set_title("Normalized (row-wise)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\nConfusion matrix saved to {cm_path}")

    # save text metrics
    metrics_path = RESULTS_DIR / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"precision_macro: {p:.4f}\n")
        f.write(f"recall_macro: {r:.4f}\n")
        f.write(f"f1_macro: {f1:.4f}\n\n")
        f.write(classification_report(true_labels, all_preds, target_names=LABEL_NAMES))
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    run_evaluation()
