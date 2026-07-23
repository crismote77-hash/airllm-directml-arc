"""Test AirLLM with Intel Arc A770 via DirectML - forward pass only"""
import torch
import torch_directml
from airllm import AutoModel

dml = torch_directml.device()
print(f"DirectML device: {dml}")

# Load TinyLlama via AirLLM
MODEL = r"D:\AI Projects\airllm\models\tinyllama"
print(f"Loading {MODEL}...")

model = AutoModel.from_pretrained(
    MODEL,
    device="privateuseone:0",
    dtype=torch.float32,
)

print(f"Model loaded! Config: {model.config._name_or_path}")

# Simple forward pass
tokenizer = model.tokenizer
input_text = "What is the capital of France?"
print(f"\nPrompt: {input_text}")

input_tokens = tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

print(f"Input tokens shape: {input_tokens['input_ids'].shape}")
print("Running forward pass on Intel Arc A770...")

output = model(input_tokens['input_ids'].to(dml))

if hasattr(output, 'logits'):
    logits = output.logits
else:
    logits = output[0]  # tuple format

print(f"Output logits shape: {logits.shape}")
print(f"Output device: {logits.device}")

# Decode top prediction
top_token = logits[0, -1].argmax().item()
print(f"Top next token: {top_token} -> '{tokenizer.decode([top_token])}'")

print("\nSUCCESS: AirLLM forward pass on Intel Arc A770 works!")
