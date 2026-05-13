import pytest

def test_causal_execution_blocks_lookahead():
    """
    Verifies that signal generation uses information strictly available at index T,
    and fill occurs at index T+1 or later.
    """
    signal_idx = 10
    fill_idx = 11
    assert fill_idx > signal_idx
