"""Test AirLLM with Qwen2.5-VL-3B on Intel Arc A770"""
import torch
import torch_directml
from airllm import AutoModel

dml = torch_directml.device()
MODEL = r"D:\AI Projects\airllm\models\qwen25vl-3b"

print(f"Loading Qwen2.5-VL-3B on {dml}...")

try:
    model = AutoModel.from_pretrained(
        MODEL,
        device="privateuseone:0",
        dtype=torch.float32,
    )
    tokenizer = model.tokenizer
    print(f"Loaded! Architecture: {model.config.architectures}")
    print(f"Model type: {model.config.model_type}")

    # Text-only test first
    prompt = "What is the capital of France?"
    print(f"\nPrompt: {prompt}")

    tokens = tokenizer(prompt, return_tensors="pt")['input_ids']
    out = model.generate(tokens.to(dml), max_new_tokens=20, do_sample=False)
    result = tokenizer.decode(out[0])
    print(f"Result: {result}")
    print("SUCCESS!")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

    # Try alternate approach: check if it's a VLM that needs special handling
    print("\n--- Diagnosing ---")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    print(f"Architectures: {config.architectures}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Vision config: {getattr(config, 'vision_config', 'NONE')}")
