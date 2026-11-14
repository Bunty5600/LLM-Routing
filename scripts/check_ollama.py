"""Environment check for the LLM escalation research project.

Usage:
    python scripts/check_ollama.py qwen2.5:3b
    python scripts/check_ollama.py qwen2.5:7b
"""
import platform
import sys
import requests

BASE = "http://localhost:11434"
PROMPT = (
    "Tom has 3 boxes with 4 apples each. He eats 2 apples. "
    "How many apples are left? End with 'Final answer: <number>'."
)


def ns_to_s(ns):
    return round(ns / 1e9, 3) if ns else None


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    # 1. Is the server running, and what version?
    try:
        v = requests.get(f"{BASE}/api/version", timeout=5).json()
        print(f"Ollama version: {v.get('version')}")
    except Exception as e:
        print(f"FAIL: cannot reach Ollama at {BASE}: {e}")
        print("Start it with: ollama serve   (or open the Ollama app)")
        return

    # 2. Which models are installed?
    tags = requests.get(f"{BASE}/api/tags", timeout=5).json().get("models", [])
    print("Installed models:")
    for m in tags:
        print(f"  {m['name']}  ({m.get('size', 0) / 1e9:.1f} GB)")

    # 3. One call, asking for logprobs
    body = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "logprobs": True,
        "options": {"temperature": 0, "seed": 1, "num_predict": 200},
    }
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=600)
    if r.status_code != 200:
        print(f"FAIL: status {r.status_code}: {r.text[:300]}")
        return
    d = r.json()

    print("\n--- Response ---")
    print(d.get("response", "").strip())

    print("\n--- Timing and tokens ---")
    print(f"load_duration (s):        {ns_to_s(d.get('load_duration'))}")
    print(f"prompt tokens:            {d.get('prompt_eval_count')}")
    print(f"prompt_eval_duration (s): {ns_to_s(d.get('prompt_eval_duration'))}")
    print(f"generated tokens:         {d.get('eval_count')}")
    print(f"eval_duration (s):        {ns_to_s(d.get('eval_duration'))}")
    ev, dur = d.get("eval_count"), d.get("eval_duration")
    if ev and dur:
        print(f"generation speed:         {ev / (dur / 1e9):.1f} tokens/s")

    print("\n--- Logprobs ---")
    lp = d.get("logprobs")
    if lp:
        print(f"Logprobs returned: YES ({len(lp)} entries)")
        print(f"First entry: {lp[0]}")
    else:
        print("Logprobs returned: NO (E2 will be skipped, or update Ollama)")

    print("\nNow run `ollama ps` in another terminal to see the GPU/CPU split.")


if __name__ == "__main__":
    main()