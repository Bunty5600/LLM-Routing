"""GSM8K loading, fixed-seed sampling and saving.

The Hugging Face `datasets` package is imported only inside load_split (the one
function that downloads data). This keeps every other module, the replay and
the tests independent of pyarrow, which Windows may block.
"""
import json
import random
from pathlib import Path


def parse_reference(answer_text):
    """Return the number after '####', with commas removed.

    Dry run: "...so she has 1,234 eggs.\n#### 1,234"
      -> split on '####' -> " 1,234" -> strip -> "1,234"
      -> remove commas -> "1234" -> float 1234.0 -> integral -> int 1234
    """
    if "####" not in answer_text:
        raise ValueError("No '####' marker in reference answer")
    final = answer_text.split("####")[-1].strip().replace(",", "")
    value = float(final)
    return int(value) if value.is_integer() else value


def load_split(split, n, seed, exclude_ids=None,
               dataset_name="openai/gsm8k", dataset_config="main"):
    """Return n questions as [{id, question, reference_answer}].

    The sample is fixed by `seed`. Ids in `exclude_ids` are removed from the
    pool BEFORE sampling, so two splits can never share a question.

    Dry run (illustrative numbers): train has 7473 rows, pool = indices 0..7472.
    Sampling 25 with seed 42 might give indices [3452, 1048, ...] -> ids
    "train-3452", "train-1048". A second call with those ids excluded samples
    50 from the remaining 7448 rows, so overlap is impossible.
    """
    from datasets import load_dataset  # lazy: needs pyarrow, only used for downloading

    ds = load_dataset(dataset_name, dataset_config, split=split)
    excluded = set(exclude_ids or [])
    pool = [i for i in range(len(ds)) if f"{split}-{i}" not in excluded]
    if n > len(pool):
        raise ValueError(f"Asked for {n} questions but only {len(pool)} available")

    rng = random.Random(seed)
    chosen = rng.sample(pool, n)

    items = []
    for i in chosen:
        row = ds[i]
        items.append({
            "id": f"{split}-{i}",
            "question": row["question"],
            "reference_answer": parse_reference(row["answer"]),
        })
    return items


def save_items(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)