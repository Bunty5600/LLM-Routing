import pytest  # type: ignore[import-not-found]

from src.evaluation.answer_extract import extract_final_number, is_correct


@pytest.mark.parametrize("text, expected", [
    ("Final answer: 10", 10.0),
    ("12 - 2 = 10. Final answer: 10.", 10.0),
    ("Final answer: $1,234.50", 1234.5),
    ("Final answer: 1,000,000", 1000000.0),
    ("Final answer: -7", -7.0),
    ("Final answer: 3.5", 3.5),
    ("Final answer: **18**", 18.0),
    ("Final answer: 10 apples", 10.0),
    ("Final answer: \\boxed{42}", 42.0),
    ("Final answer: 25%", 25.0),
    ("FINAL ANSWER: 6", 6.0),
    ("Final answer: 5. Wait, I rechecked.\nFinal answer: 8", 8.0),  # last marker wins
    ("The total is 5 and then 9", 9.0),                  # fallback: last number
    ("12 - 2 = 10. Final answer: unknown", 10.0),        # marker but no number after it
    ("3-4 apples", 4.0),                                 # hyphen is not a minus sign
])
def test_extract_numbers(text, expected):
    assert extract_final_number(text) == expected


@pytest.mark.parametrize("text", ["", "no numbers here", None])
def test_missing_number_returns_none(text):
    assert extract_final_number(text) is None


def test_is_correct_basic():
    assert is_correct(10.0, 10)
    assert is_correct(-3.0, -3)
    assert is_correct(0.1 + 0.2, 0.3)
    assert not is_correct(10.5, 10)
    assert not is_correct(11.0, 10)


def test_is_correct_missing_prediction():
    assert not is_correct(None, 10)