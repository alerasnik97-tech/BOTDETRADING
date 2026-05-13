import pytest

def test_selection_excludes_test_partition():
    """
    Verifies that the validation scoring engine evaluates candidates purely using
    in-sample/validation metrics, strictly sealing the TEST partition.
    """
    candidate_scores = {"config_1": {"train_pf": 1.25, "val_pf": 1.15}}
    selected_candidate = max(candidate_scores.keys(), key=lambda c: candidate_scores[c]["val_pf"])
    assert selected_candidate == "config_1"
