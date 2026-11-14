"""Live check: one real pilot question through adapter + extraction + cache.

Usage:  python scripts\\smoke_adapter.py            (uses small_a from config)
        python scripts\\smoke_adapter.py qwen2.5:7b
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.data.gsm8k import load_items
from src.evaluation.answer_extract import extract_final_number, is_correct
from src.models.ollama_adapter import generate, warmup


def main():
    cfg = load_config()
    model = sys.argv[1] if len(sys.argv) > 1 else cfg["models"]["small_a"]
    item = load_items(resolve("data/pilot.json"))[0]
    prompt = cfg["generation"]["prompt_template"].format(question=item["question"])

    print(f"Model: {model}")
    print(f"Question id: {item['id']}  (reference answer: {item['reference_answer']})")
    load_s = warmup(model, cfg)
    print(f"Warm-up done (load time {load_s:.1f} s)\n")

    for n in (1, 2):
        rec = generate(model, prompt,
                       temperature=cfg["generation"]["greedy_temperature"],
                       seed=cfg["seed"], logprobs=True, cfg=cfg)
        pred = extract_final_number(rec["text"])
        lps = rec["logprobs"]
        mean_lp = (sum(lps) / len(lps)) if lps else None
        print(f"--- Call {n} ---")
        print(f"from_cache:       {rec['from_cache']}")
        print(f"answer tail:      ...{rec['text'][-100:].strip()!r}")
        print(f"extracted number: {pred}")
        print(f"correct:          {is_correct(pred, item['reference_answer'])}")
        print(f"tokens:           prompt={rec['prompt_tokens']}, generated={rec['generated_tokens']}")
        print(f"compute_seconds:  {rec['compute_seconds']:.3f}  (load excluded)")
        print(f"logprobs:         {len(lps) if lps else 'none'} entries, "
              f"mean logprob = {mean_lp if mean_lp is None else round(mean_lp, 4)}\n")


if __name__ == "__main__":
    main()