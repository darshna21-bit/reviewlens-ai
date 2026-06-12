---
title: ReviewLens AI
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# ReviewLens AI — Amazon Review Intelligence Platform

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/Darshna21/reviewlens-ai)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/darshna21-bit/reviewlens-ai)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey)](https://flask.palletsprojects.com)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44-orange)](https://huggingface.co/transformers)

**Live Demo:** https://huggingface.co/spaces/Darshna21/reviewlens-ai

An end-to-end NLP system that fine-tunes **DistilBERT** for 3-class sentiment classification and **T5-small** for abstractive summarization on real Amazon customer reviews — served via a Flask REST API with a custom dark-themed web UI.

---

## What This Project Does

Customers write long, nuanced reviews that are hard to process at scale. ReviewLens AI solves this with two fine-tuned transformer models:

- **Sentiment Analysis** (DistilBERT): Classifies reviews as Positive / Neutral / Negative with per-class confidence scores
- **Abstractive Summarization** (T5-small): Distills a lengthy review into a single essential sentence — not extractive clipping, but true re-generation

Both models run in under a second on CPU, served through a clean Flask REST API with three endpoints.

---

## Dataset

**Amazon Fine Food Reviews** — [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

- **15,000 reviews** used for training (stratified train/val/test split: 70/15/15)
- Real Amazon product reviews with star ratings (1–5) and human-written headlines
- Star ratings mapped to 3-class sentiment labels:

| Stars | Label    | Class ID |
| ----- | -------- | -------- |
| 1–2   | Negative | 0        |
| 3     | Neutral  | 1        |
| 4–5   | Positive | 2        |

- `review_headline` used as reference summaries for T5 seq2seq training — no manual annotation needed
- Text cleaning: lowercasing, deduplication, removal of reviews under 20 characters

---

## Models

### DistilBERT — Sentiment Classification

- Base: `distilbert-base-uncased` with a 3-class linear classification head
- **Why 3-class over binary?** Binary classifiers confidently mislabel ambivalent 3-star reviews as "positive." Adding a Neutral class gave cleaner decision boundaries and better macro F1.
- Key training choices:
  - `DataCollatorWithPadding` (dynamic padding) → ~30% faster training vs. fixed 512-length padding
  - Learning rate: `2e-5` with 10% warmup ratio — `5e-5` caused overfitting on this dataset size
  - Early stopping (patience=2) on macro F1 to prevent overfitting
  - `fp16=True` on GPU for faster training
  - `model.eval()` + `torch.no_grad()` at inference — essential for memory efficiency on CPU

### T5-small — Abstractive Summarization

- Base: `t5-small` fine-tuned for seq2seq on (review_body → review_headline) pairs
- **Key insight:** T5's multi-task pretraining requires a `"summarize: "` prefix to activate the summarization decoder. Without it, the model copies the input verbatim.
- Beam search with `num_beams=4` and `no_repeat_ngram_size=3` — greedy decoding produced repetitive outputs on longer reviews
- `min_length=20` prevents degenerate one-word outputs like "Good." on short reviews

---

## Evaluation Results

### Sentiment (DistilBERT, 15k samples)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | ~0.84 |
| Precision (macro) | ~0.82 |
| Recall (macro)    | ~0.80 |
| F1 (macro)        | ~0.81 |

> Macro F1 is the right metric here — Amazon reviews skew heavily positive (~60% of the dataset is 4–5 star), so accuracy alone is misleading. A naive "always predict positive" baseline would score ~60% accuracy while being useless.

### Summarization (T5-small, 15k samples)

| Metric  | Value |
| ------- | ----- |
| ROUGE-1 | ~0.31 |
| ROUGE-2 | ~0.14 |
| ROUGE-L | ~0.28 |

> ROUGE scores for review summarization are lower than news benchmarks by design — review headlines are subjective and varied, unlike news headlines which closely follow article content. This is an inherent ceiling on automatic metrics, not a model failure.

---

## System Architecture

```
reviewlens-ai/
├── app/                          # Flask application
│   ├── __init__.py               # Application factory (create_app)
│   ├── api/routes.py             # Route definitions — thin handlers only
│   ├── controllers/              # Business logic, input validation
│   ├── services/
│   │   ├── sentiment_service.py  # DistilBERT inference wrapper
│   │   └── summarization_service.py  # T5 inference wrapper
│   ├── models/model_loader.py    # Singleton — loads both models once at startup
│   └── utils/text_utils.py       # Text cleaning, truncation, formatting
├── config/settings.py            # Centralized config with env variants
├── templates/                    # Jinja2 HTML templates
├── static/
│   ├── css/main.css              # Dark-themed UI (no CSS framework)
│   └── js/app.js                 # Vanilla JS — no frameworks, zero bundle size
├── training/
│   ├── sentiment/
│   │   ├── train_sentiment.py    # DistilBERT fine-tuning
│   │   └── evaluate_sentiment.py # Metrics + confusion matrix
│   └── summarization/
│       ├── train_summarization.py    # T5-small fine-tuning
│       └── evaluate_summarization.py # ROUGE evaluation
├── tests/                        # pytest suite (mocked models, no weights needed)
├── Dockerfile                    # Docker deployment
└── wsgi.py                       # Gunicorn entrypoint
```

**Design decisions:**

- Models loaded once at startup via a singleton `ModelLoader` — avoids 3s cold-start latency on first request
- Services are plain classes (not Flask globals) — easy to mock in tests without loading 250MB weights
- No JavaScript frameworks — vanilla `fetch` API + DOM manipulation kept bundle size at zero
- Environment-based config (`DevelopmentConfig` / `ProductionConfig`) — model paths configurable via env vars, no hardcoded paths

---

## API Documentation

All endpoints prefixed with `/api/`.

### GET /api/health

```bash
curl https://huggingface.co/spaces/Darshna21/reviewlens-ai/api/health
```

```json
{
  "status": "ok",
  "models": {
    "sentiment_model": "loaded",
    "summarization_model": "loaded",
    "device": "cpu"
  }
}
```

### POST /api/predict — Sentiment Only

```bash
curl -X POST /api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is excellent, works perfectly and arrived fast."}'
```

```json
{
  "status": "ok",
  "result": {
    "label": "Positive",
    "confidence": 0.9952,
    "scores": [
      { "label": "Negative", "score": 0.0021 },
      { "label": "Neutral", "score": 0.0027 },
      { "label": "Positive", "score": 0.9952 }
    ]
  }
}
```

### POST /api/summarize — Summary Only

```bash
curl -X POST /api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "I bought this coffee maker three months ago..."}'
```

```json
{
  "status": "ok",
  "result": {
    "summary": "Great coffee maker that transformed my morning routine.",
    "input_length": 48,
    "summary_length": 9
  }
}
```

### POST /api/analyze — Combined (Sentiment + Summary)

```bash
curl -X POST /api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your full review here..."}'
```

Returns both sentiment classification and abstractive summary in a single response.

---

## Running Locally

```bash
# 1. Clone and set up
git clone https://github.com/darshna21-bit/reviewlens-ai.git
cd reviewlens-ai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 2. Download dataset (requires free Kaggle account)
pip install kaggle
kaggle datasets download -d snap/amazon-fine-food-reviews -p data/raw/ --unzip

# 3. Preprocess
python training/data_pipeline.py --max-rows 15000

# 4. Train DistilBERT
python training/sentiment/train_sentiment.py

# 5. Train T5-small
python training/summarization/train_summarization.py

# 6. Evaluate
python training/sentiment/evaluate_sentiment.py
python training/summarization/evaluate_summarization.py

# 7. Start server
python app.py
# → http://localhost:5000

# 8. Run tests
pytest tests/ -v
```

---

## Key Challenges & What I Learned

**Class imbalance:** Amazon reviews skew heavily positive. I initially hit 84% accuracy and thought training was going well — until I checked the confusion matrix and saw the model was predicting "Positive" for most neutral reviews. Switching from accuracy to macro F1 as the training metric and adding a third Neutral class fixed the problem and made the evaluation honest.

**T5 task prefixes:** T5's multi-task pretraining means it needs a text prefix (`"summarize: "`) to activate the right decoder path. Without it, the model copies the input. This cost me significant debugging time and is a non-obvious gotcha when fine-tuning T5 for the first time.

**CPU inference memory:** Loading both DistilBERT and T5-small simultaneously on CPU requires careful memory management. `model.eval()` disables dropout (slightly reduces memory), and `torch.no_grad()` prevents building a computation graph during inference — without it, PyTorch doubles memory usage for every forward pass.

**Singleton model loading:** Initially each service loaded its own model, causing ~3s latency on the first request to each endpoint. Moving to startup loading via a shared singleton made first-request latency fast and kept memory usage predictable.

---

## Future Improvements

- **Aspect-based sentiment:** Identify sentiment per product attribute (battery, build quality, screen) rather than a single review-level label
- **Model calibration:** Fine-tuned classifiers aren't well-calibrated — temperature scaling would make confidence scores more meaningful
- **Parallel inference:** The `/analyze` endpoint runs both models sequentially; parallelizing would halve end-to-end latency
- **Streaming summarization:** Stream T5 token generation to the UI for better perceived performance on long reviews
- **Better summarization targets:** `review_headline` sometimes contains uninformative text like "Five Stars" — filtering these during preprocessing would improve ROUGE meaningfully

---

## Tech Stack

| Component           | Technology                              |
| ------------------- | --------------------------------------- |
| Sentiment Model     | DistilBERT (distilbert-base-uncased)    |
| Summarization Model | T5-small                                |
| Framework           | 🤗 Transformers 4.44, PyTorch           |
| Backend             | Flask + Gunicorn                        |
| Frontend            | Vanilla HTML/CSS/JS (zero dependencies) |
| Deployment          | Docker → Hugging Face Spaces            |
| Training            | Google Colab (T4 GPU)                   |

---

_Built by **Darshna Shingavi**, 3rd Year CSE student. Feedback welcome via [GitHub Issues](https://github.com/darshna21-bit/reviewlens-ai/issues)._
