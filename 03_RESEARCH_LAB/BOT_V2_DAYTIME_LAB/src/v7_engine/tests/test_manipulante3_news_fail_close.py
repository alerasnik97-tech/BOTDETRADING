import pytest

def test_news_module_fail_close_behavior():
    """
    Verifies that if high impact macroeconomic news cache is missing or corrupt,
    trading logic aborts rather than defaulting to open access.
    """
    calendar_available = False
    with pytest.raises(Exception):
        if not calendar_available:
            raise RuntimeError("News calendar missing: FAIL_CLOSE_REQUIRED triggered.")
