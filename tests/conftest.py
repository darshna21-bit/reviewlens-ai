"""
Shared fixtures for the test suite.

Tests use TestingConfig which:
- Points model paths to non-existent dirs (models won't load)
- Sets TESTING=True so _load_models() is skipped in create_app
- Enables Flask test client
"""

import pytest
from unittest.mock import MagicMock, patch

from app import create_app
from config.settings import TestingConfig


@pytest.fixture(scope="session")
def app():
    """
    App instance shared across the session. Session scope is fine here because
    none of the tests mutate app state — if they did, function scope would be safer.
    """
    application = create_app(config_override=TestingConfig)
    yield application


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture
def sample_positive_review():
    return (
        "This product completely exceeded my expectations. The build quality is "
        "exceptional and it works exactly as advertised. Would definitely buy again "
        "and highly recommend to anyone looking for a reliable option."
    )


@pytest.fixture
def sample_negative_review():
    return (
        "Terrible quality and a complete waste of money. Broke after two days of "
        "normal use. Customer service was unhelpful and refused a refund. "
        "Stay far away from this product."
    )


@pytest.fixture
def sample_neutral_review():
    return (
        "The product is okay. It does what it says it does, nothing more nothing less. "
        "Arrived on time and packaging was fine. I might consider alternatives next time "
        "but it served the purpose for now."
    )


@pytest.fixture
def mock_sentiment_result():
    return {
        "label": "Positive",
        "confidence": 0.9234,
        "scores": [
            {"label": "Negative", "score": 0.0321},
            {"label": "Neutral",  "score": 0.0445},
            {"label": "Positive", "score": 0.9234},
        ],
    }


@pytest.fixture
def mock_summary_result():
    return {
        "summary": "Great product that exceeded expectations. Highly recommended.",
        "input_length": 42,
        "summary_length": 9,
    }
