"""Check which packages can be imported (read-only, changes nothing).

Run with your main Python, then with the routellm environment:

    python scripts\\check_imports.py
    ..\\venv-routellm\\Scripts\\python.exe scripts\\check_imports.py
"""
import importlib
import sys

print("python", sys.version.split()[0], "|", sys.executable)
for name in ("numpy", "scipy.stats", "pandas", "pyarrow", "datasets",
             "matplotlib.pyplot", "yaml", "requests"):
    try:
        importlib.import_module(name)
        print(f"{name:<20} OK")
    except Exception as e:
        print(f"{name:<20} FAILED  {type(e).__name__}: {str(e)[:90]}")