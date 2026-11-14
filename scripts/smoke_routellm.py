"""Smoke test: load RouteLLM's BERT router and score a few pilot questions.

Run with the routellm environment (NOT your main Python):

    ..\\venv-routellm\\Scripts\\python.exe scripts\\smoke_routellm.py

The first run downloads the router checkpoint from Hugging Face (needs
internet), so the 'load' time of that first run includes the download.
The scores are only printed, not saved.

A placeholder (NOT a real key) is set for this process only, because
importing RouteLLM creates an OpenAI client at import time. The BERT router
itself does not call OpenAI.

Dry run of what a score means (illustrative numbers): classifier outputs
[2.0, 0.5, -1.0] -> probabilities [0.786, 0.175, 0.039] -> the router returns
1 - (0.175 + 0.039) = 0.786. Higher = the strong model is more likely needed.
"""
import os

os.environ["OPENAI_API_KEY"] = "placeholder-not-a-real-key"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "configs" / "experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rcfg = cfg["routellm"]

    with open(ROOT / cfg["data"]["dir"] / "pilot.json", "r", encoding="utf-8") as f:
        items = json.load(f)[: rcfg["smoke_questions"]]

    print(f"Loading router '{rcfg['router']}' from checkpoint {rcfg['checkpoint']} ...")
    t0 = time.perf_counter()
    try:
        from routellm.routers.routers import BERTRouter
        router = BERTRouter(rcfg["checkpoint"])
    except Exception as e:
        print(f"\nFAILED to load the router: {type(e).__name__}: {e}")
        print("If this is a network error, check your internet and run it again.")
        sys.exit(1)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in router.model.parameters()) / 1e6
    print(f"Router loaded in {load_s:.1f} s ({n_params:.1f} M parameters, CPU)\n")

    print(f"{'question id':<14}{'score':>8}{'seconds':>10}")
    for i, item in enumerate(items):
        t = time.perf_counter()
        score = router.calculate_strong_win_rate(item["question"])
        dt = time.perf_counter() - t
        tag = "  (first call)" if i == 0 else ""
        print(f"{item['id']:<14}{score:>8.3f}{dt:>10.3f}{tag}")


if __name__ == "__main__":
    main()