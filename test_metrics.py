import pytest
from metrics import anls_score, relaxed_acc_score, boolean_acc_score, exact_match

def test_anls():
    assert anls_score("hello world", ["hello world"]) == 1.0
    assert anls_score("hello world", ["completely different"]) == 0.0
    
    # 1 substitution out of 5 chars = dist 1. Score = 1 - 1/5 = 0.8
    assert anls_score("apple", ["apply"]) == 0.8
    
    # Below threshold of 0.5 (e.g. score 0.4) -> should be 0.0
    # "abc" vs "abcdefg" -> dist 4, max_len 7 -> score = 1 - 4/7 = 3/7 = 0.428 < 0.5
    assert anls_score("abc", ["abcdefg"]) == 0.0

def test_relaxed_acc():
    # 5% boundary
    assert relaxed_acc_score("100", ["105"]) == 1.0  # (105-100)/105 = 0.047 <= 0.05
    assert relaxed_acc_score("100", ["106"]) == 0.0  # (106-100)/106 = 0.056 > 0.05
    
    # Percentages and commas
    assert relaxed_acc_score("1,000.5%", ["1000.5"]) == 1.0
    
    # Exact text match
    assert relaxed_acc_score("Red", ["red"]) == 1.0
    
    # Zero handling
    assert relaxed_acc_score("0", ["0"]) == 1.0
    assert relaxed_acc_score("0.1", ["0"]) == 0.0

def test_boolean_acc():
    assert boolean_acc_score("True", ["True"]) == 1.0
    assert boolean_acc_score("yes", ["True"]) == 1.0
    assert boolean_acc_score("False", ["False"]) == 1.0
    assert boolean_acc_score("no", ["False"]) == 1.0
    
    assert boolean_acc_score("yes", ["False"]) == 0.0
    assert boolean_acc_score("potato", ["True"]) == 0.0
    assert boolean_acc_score("True", ["potato"]) == 0.0
