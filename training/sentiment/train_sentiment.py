"""
Fine-tunes DistilBERT for 3-class sentiment classification.

Training on full dataset takes ~45min on a T4 GPU (Colab) or ~3-4 hours on
CPU. Use --max-train-samples 5000 for a quick sanity check run.

Run: python training/sentiment/train_sentiment.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_OUTPUT_DIR = Path("saved_models/sentiment")
RESULTS_DIR = Path("training/sentiment/results")
BASE_MODEL = "distilbert-base-uncased"
NUM_LABELS = 3

# id2label and label2id need to be set in the model config so that
# from_pretrained() reconstructs them correctly during inference
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}


def load_data(max_train_samples: int | None = None) -> tuple[Dataset, Dataset, Dataset]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    if max_train_samples:
        # stratified subsample so we keep class balance even in tiny runs
        train_df = (
            train_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_train_samples // 3), random_state=42))
        )
        logger.info(f"Using {len(train_df)} training samples (max_train_samples={max_train_samples})")

    # HuggingFace Datasets expects "label" as the target column name
    train_ds = Dataset.from_pandas(train_df[["review_body", "label"]].rename(columns={"review_body": "text"}))
    val_ds = Dataset.from_pandas(val_df[["review_body", "label"]].rename(columns={"review_body": "text"}))
    test_ds = Dataset.from_pandas(test_df[["review_body", "label"]].rename(columns={"review_body": "text"}))

    logger.info(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def tokenize_dataset(dataset: Dataset, tokenizer: DistilBertTokenizerFast) -> Dataset:
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            # not padding here — DataCollatorWithPadding handles dynamic
            # padding per batch, which is more memory efficient than
            # padding everything to 512 upfront
            padding=False,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
    }


def train(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    logger.info("Loading and tokenizing datasets...")
    train_ds, val_ds, test_ds = load_data(max_train_samples=args.max_train_samples)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)

    logger.info(f"Initializing DistilBERT with {NUM_LABELS} output labels...")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # fp16 on GPU saves memory and speeds up training significantly;
    # set to False on CPU to avoid errors
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_dir=str(Path("logs") / "sentiment"),
        logging_steps=50,
        save_total_limit=2,  # only keep the 2 best checkpoints to save disk space
        report_to="none",    # set to "wandb" if you want W&B tracking
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    logger.info(f"Test results: {test_results}")

    # save final model + tokenizer to the clean output dir (not the checkpoints dir)
    logger.info(f"Saving model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # save results for the README table
    results_path = RESULTS_DIR / "test_metrics.txt"
    with open(results_path, "w") as f:
        for k, v in test_results.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Test metrics saved to {results_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment analysis")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples for a quick run (e.g. 5000)"
    )
    args = parser.parse_args()
    train(args)
