import logging

from app.services.sentiment_service import sentiment_service
from app.services.summarization_service import summarization_service
from app.utils.text_utils import validate_input
from config.settings import get_config

logger = logging.getLogger(__name__)
cfg = get_config()


class AnalysisController:
    """
    Sits between routes and services. Handles input validation, error
    classification, and combining results from multiple services.

    Separating this from the route handlers keeps the routes thin and makes
    the business logic unit-testable without spinning up Flask.
    """

    def analyze_sentiment(self, text: str) -> tuple[dict, int]:
        """Returns (response_dict, http_status_code)"""
        valid, error_msg = validate_input(text, min_length=10, max_length=cfg.MAX_INPUT_LENGTH)
        if not valid:
            return {"error": error_msg}, 400

        try:
            result = sentiment_service.predict(text)
            return {"status": "ok", "result": result}, 200
        except RuntimeError as e:
            # model not loaded — service unavailable rather than internal error
            logger.warning(f"Sentiment service unavailable: {e}")
            return {"error": str(e)}, 503
        except Exception as e:
            logger.exception(f"Unexpected error during sentiment analysis: {e}")
            return {"error": "Internal model error. Check server logs."}, 500

    def analyze_summarization(self, text: str) -> tuple[dict, int]:
        valid, error_msg = validate_input(text, min_length=50, max_length=cfg.MAX_INPUT_LENGTH)
        if not valid:
            # slightly stricter minimum for summarization — trying to summarize
            # a 10-word review doesn't make much sense
            if len((text or "").strip()) < 50:
                return {
                    "error": "Review text must be at least 50 characters for summarization."
                }, 400
            return {"error": error_msg}, 400

        try:
            result = summarization_service.summarize(text)
            return {"status": "ok", "result": result}, 200
        except RuntimeError as e:
            logger.warning(f"Summarization service unavailable: {e}")
            return {"error": str(e)}, 503
        except Exception as e:
            logger.exception(f"Unexpected error during summarization: {e}")
            return {"error": "Internal model error. Check server logs."}, 500

    def analyze_combined(self, text: str) -> tuple[dict, int]:
        """
        Runs both sentiment and summarization in sequence.
        If one model is unavailable we still return what we can rather than
        failing the whole request — partial results are more useful than nothing.
        """
        valid, error_msg = validate_input(text, min_length=10, max_length=cfg.MAX_INPUT_LENGTH)
        if not valid:
            return {"error": error_msg}, 400

        response = {"status": "ok", "result": {}}
        has_error = False

        # sentiment
        try:
            sentiment_result = sentiment_service.predict(text)
            response["result"]["sentiment"] = sentiment_result
        except RuntimeError as e:
            response["result"]["sentiment"] = {"error": str(e)}
            has_error = True
        except Exception as e:
            logger.exception(f"Sentiment error in combined analysis: {e}")
            response["result"]["sentiment"] = {"error": "Model inference failed"}
            has_error = True

        # summarization — only meaningful for longer texts
        if len(text.split()) >= 20:
            try:
                summary_result = summarization_service.summarize(text)
                response["result"]["summarization"] = summary_result
            except RuntimeError as e:
                response["result"]["summarization"] = {"error": str(e)}
                has_error = True
            except Exception as e:
                logger.exception(f"Summarization error in combined analysis: {e}")
                response["result"]["summarization"] = {"error": "Model inference failed"}
                has_error = True
        else:
            response["result"]["summarization"] = {
                "skipped": True,
                "reason": "Text too short for meaningful summarization (< 20 words)"
            }

        if has_error:
            response["status"] = "partial"

        return response, 200


analysis_controller = AnalysisController()
