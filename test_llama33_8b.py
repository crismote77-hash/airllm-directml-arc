"""AirLLM + DirectML smoke test for community Llama 3.3 8B."""
import os
import time
from pathlib import Path

import torch
import torch_directml
from airllm import AutoModel

MODEL = Path(os.environ.get("AIRLLM_MODEL_PATH", "models/llama33-8b")).resolve()
DEVICE = "privateuseone:0"

if not (MODEL / "config.json").is_file():
    raise SystemExit(
        "Local checkpoint not found. Set AIRLLM_MODEL_PATH to a downloaded "
        "allura-forge/Llama-3.3-8B-Instruct checkout. This script never downloads models."
    )

print(f"torch={torch.__version__} device={torch_directml.device()}")
started = time.perf_counter()
model = AutoModel.from_pretrained(str(MODEL), device=DEVICE, dtype=torch.float32)
print(f"LOAD_SECONDS={time.perf_counter() - started:.2f}")
print(f"ARCHITECTURE={model.config.architectures}")

prompt = "The capital of France is"
encoded = model.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)

started = time.perf_counter()
out = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=4,
    do_sample=False,
)
elapsed = time.perf_counter() - started
text = model.tokenizer.decode(out[0], skip_special_tokens=True)
print(f"GENERATION_SECONDS={elapsed:.2f}")
print(f"RESULT={text}")
assert "Paris" in text, text
print("SUCCESS: Llama-3.3-8B generated correctly on Intel Arc DirectML")
