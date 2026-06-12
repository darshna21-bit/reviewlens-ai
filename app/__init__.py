import logging
import os
from pathlib import Path

from flask import Flask

from config.settings import get_config


def create_app(config_override=None) -> Flask:
    """
    Application factory pattern. Makes testing easier because each test
    can spin up its own app instance with TestingConfig rather than
    sharing global state.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / "templates"),
        static_folder=str(Path(__file__).parent.parent / "static"),
    )

    # load config
    cfg = config_override or get_config()
    app.config.from_object(cfg)

    # set up logging before anything else so startup model loading is logged
    _configure_logging(app)

    # register blueprints
    from app.api.routes import api_bp, main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # load models at startup — doing this inside the factory so tests can
    # skip it by pointing config to non-existent paths
    _load_models(app)

    return app


def _configure_logging(app: Flask):
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"), logging.INFO)
    log_dir = app.config.get("LOG_DIR", "logs")
    Path(log_dir).mkdir(exist_ok=True)

    handlers = [logging.StreamHandler()]

    # file handler in production — helpful for debugging on Render
    if not app.config.get("TESTING"):
        file_handler = logging.FileHandler(Path(log_dir) / "app.log")
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _load_models(app: Flask):
    from app.models.model_loader import model_loader

    sentiment_path = app.config.get("SENTIMENT_MODEL_PATH", "")
    summarization_path = app.config.get("SUMMARIZATION_MODEL_PATH", "")

    if not app.config.get("TESTING"):
        model_loader.load_sentiment_model(sentiment_path)
        model_loader.load_summarization_model(summarization_path)
