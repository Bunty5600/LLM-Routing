"""Create and save the pilot, calibration and evaluation splits.

Usage:  python scripts\\prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.data.gsm8k import load_split, save_items


def main():
    cfg = load_config()
    d = cfg["data"]
    seed = cfg["seed"]
    kw = {"dataset_name": d["hf_name"], "dataset_config": d["hf_config"]}

    pilot = load_split("train", d["pilot_size"], seed, **kw)
    pilot_ids = {x["id"] for x in pilot}

    calib = load_split("train", d["calibration_size"], seed + 1,
                       exclude_ids=pilot_ids, **kw)
    calib_ids = {x["id"] for x in calib}

    evals = load_split("test", d["evaluation_size"], seed + 2, **kw)
    eval_ids = {x["id"] for x in evals}

    # Leakage checks: by id and by question text.
    assert not (pilot_ids & calib_ids), "pilot and calibration overlap"
    assert not (pilot_ids & eval_ids), "pilot and evaluation overlap"
    assert not (calib_ids & eval_ids), "calibration and evaluation overlap"
    q = lambda items: {x["question"] for x in items}
    assert not (q(pilot) & q(calib)), "duplicate question text: pilot/calibration"
    assert not (q(calib) & q(evals)), "duplicate question text: calibration/evaluation"
    assert not (q(pilot) & q(evals)), "duplicate question text: pilot/evaluation"

    out = resolve(d["dir"])
    save_items(pilot, out / "pilot.json")
    save_items(calib, out / "calibration.json")
    save_items(evals, out / "evaluation.json")

    print(f"pilot:       {len(pilot)} questions (train)")
    print(f"calibration: {len(calib)} questions (train)")
    print(f"evaluation:  {len(evals)} questions (test)")
    print("Leakage checks passed.")
    ex = pilot[0]
    print(f"\nExample id: {ex['id']}")
    print(f"Question:   {ex['question'][:120]}...")
    print(f"Reference:  {ex['reference_answer']}")


if __name__ == "__main__":
    main()