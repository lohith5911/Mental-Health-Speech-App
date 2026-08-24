"""Verify the ML environment and key package versions required for the speech emotion pipeline."""

from __future__ import annotations

import platform
from importlib import import_module

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("matplotlib", "matplotlib"),
    ("torch", "torch"),
]


def print_version(label: str, module_name: str) -> None:
    module = import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{label}: {version}")


if __name__ == "__main__":
    print(f"Python: {platform.python_version()}")
    for label, module_name in REQUIRED_PACKAGES:
        print_version(label, module_name)

    torch = import_module("torch")
    print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")

    print("\nEnvironment verification complete.")
