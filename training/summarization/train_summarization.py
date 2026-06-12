"""
Fine-tunes T5-small for abstractive review summarization.

Uses review_headline as the ground truth summary (Amazon's own editorial
summaries are surprisingly good targets). Falls back to the first sentence
of the review if no headline is available.

Run: python training/summarization/train_summarization.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
import evaluate
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.utils.text_utils import clean_review_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_OUTPUT_DIR = Path("saved_models/summarization")
RESULTS_DIR = Path("training/summarization/results")
BASE_MODEL = "t5-small"
TASK_PREFIX = "summarize: "

# T5 tokenization constants
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64


def get_first_sentence(text: str) -> str:
    """Fallback summary: first sentence if no review_headline."""
    sentences = text.split(".")
    return sentences[0].strip() + "." if sentences else text[:100]


def load_data(max_samples: int | None = None) -> tuple[Dataset, Dataset, Dataset]:
    def load_split(filename: str) -> pd.DataFrame:
        df = pd.read_csv(PROCESSED_DIR / filename)
        df["review_body"] = df["review_body"].apply(clean_review_text)

        # if we have review_headline, use it as the summary target
        if "review_headline" in df.columns:
            df["review_headline"] = df["review_headline"].fillna("")
            df["summary_target"] = df.apply(
                lambda row: clean_review_text(row["review_headline"])
                if len(str(row["review_headline"]).strip()) > 5
                else get_first_sentence(row["review_body"]),
                axis=1,
            )
        else:
            # no headline column — use the first sentence as a weak proxy
            # this is a degraded signal but still trains the model to compress
            logger.warning("No review_headline column found, using first-sentence proxy targets.")
            df["summary_target"] = df["review_body"].apply(get_first_sentence)

        # filter out rows where the body is shorter than the summary — those
        # are data quality issues (e.g. "Great product" as the body)
        df = df[df["review_body"].str.split().str.len() > 15]
        df = df[df["summary_target"].str.len() > 5]

        return df[["review_body", "summary_target"]]

    train_df = load_split("train.csv")
    val_df = load_split("val.csv")
    test_df = load_split("test.csv")

    if max_samples:
        train_df = train_df.sample(min(len(train_df), max_samples), random_state=42)

    logger.info(f"Summarization dataset — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_ds, val_ds, test_ds


def tokenize_dataset(dataset: Dataset, tokenizer: T5TokenizerFast) -> Dataset:
    def tokenize_fn(batch):
        # prepend the task prefix so T5 knows what operation to perform
        inputs = [TASK_PREFIX + text for text in batch["review_body"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )

        # tokenizing the labels (targets) separately with text_target
        labels = tokenizer(
            text_target=batch["summary_target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["review_body", "summary_target"],
    )


def compute_rouge_metrics(tokenizer):
    """Returns a metrics function that uses the ROUGE evaluator."""
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # replace -100 (padding label) with pad token id before decoding
        # -100 is the ignore index used by the cross-entropy loss
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # strip leading/trailing whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {
            "rouge1": round(results["rouge1"], 4),
            "rouge2": round(results["rouge2"], 4),
            "rougeL": round(results["rougeL"], 4),
        }

    return compute_metrics


def train(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = T5TokenizerFast.from_pretrained(BASE_MODEL)

    logger.info("Loading and tokenizing datasets...")
    train_ds, val_ds, test_ds = load_data(max_samples=args.max_train_samples)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)

    logger.info(f"Loading T5-small from HuggingFace Hub...")
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    use_fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=args.lr,
        predict_with_generate=True,   # needed for ROUGE evaluation during training
        generation_max_length=MAX_TARGET_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        # gradient checkpointing to reduce memory — T5-small fits without it
        # but I'll keep this here for when people want to try T5-base
        gradient_checkpointing=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,  # mask padding in labels so loss ignores them
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    logger.info(f"Test ROUGE scores: {test_results}")

    logger.info(f"Saving model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # generate some sample summaries for qualitative inspection
    _save_sample_summaries(model, tokenizer, test_ds)

    metrics_path = RESULTS_DIR / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in test_results.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Training complete.")


def _save_sample_summaries(model, tokenizer, test_ds, n=10):
    """
    Writes n sample (input → generated_summary) pairs to a file.
    Useful for quick sanity checking without running the full eval script.
    """
    samples_path = RESULTS_DIR / "sample_summaries.txt"
    model.eval()
    device = next(model.parameters()).device

    with open(samples_path, "w") as f:
        for i in range(min(n, len(test_ds))):
            input_ids = torch.tensor([test_ds[i]["input_ids"]]).to(device)
            attention_mask = torch.tensor([test_ds[i]["attention_mask"]]).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True,
                )

            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            original_ids = [x if x != -100 else tokenizer.pad_token_id for x in test_ds[i]["labels"]]
            original = tokenizer.decode(original_ids, skip_special_tokens=True)

            f.write(f"Sample {i+1}\n")
            f.write(f"Target:    {original}\n")
            f.write(f"Generated: {generated}\n")
            f.write("-" * 60 + "\n")

    logger.info(f"Sample summaries saved to {samples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for review summarization")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    train(args)
