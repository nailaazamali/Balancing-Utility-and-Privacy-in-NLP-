from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import datasets
import sentence_transformers
import spacy
import torch
import transformers


out = Path("outputs/environment")
out.mkdir(parents=True, exist_ok=True)
info = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "spacy": spacy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
(out / "runtime_environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    capture_output=True,
    text=True,
    check=False,
).stdout
(out / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
print(out)
