"""Quick test: AirLLM v3.0.1 + DirectML on Intel Arc A770"""
import torch, torch_directml
from airllm import AutoModel

dml = torch_directml.device()
MODEL = r"D:\AI Projects\airllm\models\tinyllama"

print("Loading model on", dml)
model = AutoModel.from_pretrained(MODEL, device="privateuseone:0", dtype=torch.float32)

tokenizer = model.tokenizer
print(f"Loaded: {model.config._name_or_path}")

# Test generate() - v3.0.1 delegates to inner model
input_text = "The capital of France is"
print(f"Prompt: {input_text}")

tokens = tokenizer(input_text, return_tensors="pt")['input_ids']

try:
    out = model.generate(tokens.to(dml), max_new_tokens=10, do_sample=False)
    result = tokenizer.decode(out[0])
    print(f"Result: {result}")
    print("SUCCESS: generate() works on Arc A770!")
except Exception as e:
    print(f"generate() failed: {e}")
    # Fall back to forward
    out = model(tokens.to(dml))
    print(f"Forward logits shape: {out.logits.shape}, device: {out.logits.device}")
    print("Forward pass works on Arc A770")
