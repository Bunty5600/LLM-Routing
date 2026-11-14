"""Inspect the installed RouteLLM package. Run it with the routellm environment:

    ..\\venv-routellm\\Scripts\\python.exe scripts\\inspect_routellm.py

It only reads code. It does not load or download any model, and it does not
contact OpenAI. A placeholder (NOT a real key) is set only for this process,
because importing RouteLLM creates an OpenAI client at import time.
"""
import os

# Placeholder so the import succeeds. Overwrites any real key in this process.
os.environ["OPENAI_API_KEY"] = "placeholder-not-a-real-key"

import importlib.metadata as md  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402


def main():
    print("python", sys.version.split()[0])
    for pkg in ("routellm", "torch", "transformers", "huggingface-hub"):
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")

    import torch
    print("torch CUDA available:", torch.cuda.is_available())

    try:
        from routellm.routers import routers as R
    except Exception as e:  # show the real error instead of hiding it
        print(f"\nCould not import routellm.routers.routers: {type(e).__name__}: {e}")
        return

    print("\nrouters registered:", sorted(getattr(R, "ROUTER_CLS", {})))

    print("\n--- every line in routers.py that mentions OpenAI or embeddings ---")
    for i, line in enumerate(inspect.getsource(R).splitlines(), 1):
        low = line.lower()
        if "openai" in low or "embedding" in low:
            print(f"{i:>4}: {line}")

    print("\n--- abstract Router class ---")
    print(inspect.getsource(R.Router))

    cls = getattr(R, "ROUTER_CLS", {}).get("bert") or getattr(R, "BERTRouter", None)
    if cls is None:
        print("\nNo 'bert' router found in this version.")
        return
    print("\n--- BERT router ---")
    print(inspect.getsource(cls))
    print("calculate_strong_win_rate signature:",
          inspect.signature(cls.calculate_strong_win_rate))


if __name__ == "__main__":
    main()