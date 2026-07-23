"""Test AirLLM with Intel Arc A770 - full text generation"""
import torch
import torch_directml
from airllm import AutoModel

dml = torch_directml.device()
print(f"DirectML device: {dml}")

MODEL = r"D:\AI Projects\airllm\models\tinyllama"
print(f"Loading {MODEL}...")

model = AutoModel.from_pretrained(
    MODEL,
    device="privateuseone:0",
    dtype=torch.float32,
)
tokenizer = model.tokenizer
print(f"Model loaded! Config: {model.config._name_or_path}")

# Text generation helper
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """Generate text one token at a time using the forward pass."""
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'][0].tolist()
    print(f"Input: {prompt}")
    print(f"Initial tokens: {len(input_ids)}")
    
    generated = []
    
    for step in range(max_new_tokens):
        seq = torch.tensor([input_ids + generated], dtype=torch.long).to(dml)
        
        output = model(seq)
        logits = output.logits if hasattr(output, 'logits') else output[0]
        
        # Last token's logits
        next_token_logits = logits[0, -1]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()
        
        generated.append(next_token)
        token_str = tokenizer.decode([next_token])
        
        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            print(f"\n[EOS at step {step + 1}]")
            break
        
        # Print progress periodically
        if step % 5 == 0 or step < 3:
            print(f"  step {step + 1}: token={next_token} '{token_str}'")
    
    full_output = tokenizer.decode(input_ids + generated)
    return full_output

# Test prompts
prompts = [
    "What is the capital of France?",
    "The color of the sky is",
    "1 + 1 =",
]

for prompt in prompts:
    print(f"\n{'=' * 50}")
    result = generate_text(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7)
    print(f"\nResult: {result}")

print(f"\n{'=' * 50}")
print("SUCCESS: Text generation on Intel Arc A770 works!")
