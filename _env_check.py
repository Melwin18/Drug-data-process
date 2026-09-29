"""Temporary environment probe: reports Python version and installed web/ML deps."""
import importlib.util
import sys

print("python:", sys.version.split()[0])

for name in ("sklearn", "joblib", "pandas", "numpy", "flask", "fastapi", "uvicorn"):
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
