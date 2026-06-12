import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-prod")
    DEBUG = False
    TESTING = False

    # Model paths — keeping these configurable via env so I can point to
    # different checkpoints without touching code during experiments
    SENTIMENT_MODEL_PATH = os.getenv(
        "SENTIMENT_MODEL_PATH",
        str(BASE_DIR / "saved_models" / "sentiment")
    )
    SUMMARIZATION_MODEL_PATH = os.getenv(
        "SUMMARIZATION_MODEL_PATH",
        str(BASE_DIR / "saved_models" / "summarization")
    )

    # 512 is DistilBERT's hard limit anyway, but capping here lets us fail
    # gracefully before it hits the tokenizer
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = str(BASE_DIR / "logs")

    # Inference settings — these ended up mattering a lot for summarization
    # quality; T5 with beam_size=1 (greedy) produces noticeably worse summaries
    SUMMARIZATION_MAX_NEW_TOKENS = 128
    SUMMARIZATION_MIN_LENGTH = 20
    SUMMARIZATION_NUM_BEAMS = 4
    SUMMARIZATION_LENGTH_PENALTY = 2.0

    SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    # Point to tiny dummy models during tests so we don't load 250MB checkpoints
    SENTIMENT_MODEL_PATH = "tests/fixtures/mock_sentiment"
    SUMMARIZATION_MODEL_PATH = "tests/fixtures/mock_summarization"


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
