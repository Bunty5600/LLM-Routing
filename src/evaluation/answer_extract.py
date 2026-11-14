"""Turn model text into a number, and compare it with the reference."""
import math
import re

# A number not glued to a letter/digit/dot before it. Allows a leading minus.
# The lookbehind stops "3-4" from being read as 3 and -4.
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_FINAL = re.compile(r"final\s+answer\s*[:\-=]?", re.IGNORECASE)


def _strip_commas(text):
    """Remove thousands separators: '1,234,567' -> '1234567'."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _numbers(text):
    return _NUMBER.findall(_strip_commas(text))


def extract_final_number(text):
    """Return the final numeric answer as a float, or None if there is none.

    Rule 1: find the LAST 'Final answer' marker and take the first number
            in the 80 characters after it.
    Rule 2 (fallback): take the last number anywhere in the text.

    Dry run A: "12 - 2 = 10. Final answer: 10."
      marker found -> tail " 10." -> numbers ['10'] -> 10.0

    Dry run B: "The total is 5 and then 9"
      no marker -> numbers ['5', '9'] -> last one -> 9.0

    Dry run C: "Final answer: $1,234.50"
      tail " $1,234.50" -> commas removed -> " $1234.50" -> ['1234.50'] -> 1234.5
    """
    if not text:
        return None
    matches = list(_FINAL.finditer(text))
    if matches:
        tail = text[matches[-1].end():][:80]
        nums = _numbers(tail)
        if nums:
            return float(nums[0])
    nums = _numbers(text)
    return float(nums[-1]) if nums else None


def is_correct(pred, ref, abs_tol=1e-4):
    """True if pred and ref are numerically equal within a tiny tolerance.

    A missing prediction (None) counts as wrong.
    Dry run: is_correct(10.0, 10) -> True; is_correct(None, 10) -> False;
             is_correct(10.5, 10) -> False.
    """
    if pred is None or ref is None:
        return False
    return math.isclose(float(pred), float(ref), rel_tol=1e-9, abs_tol=abs_tol)