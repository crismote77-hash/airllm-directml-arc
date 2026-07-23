"""System-agnostic AirLLM manual text-generation smoke test."""

from __future__ import annotations

import argparse

from runtime_utils import (
    add_common_args,
    import_torch,
    load_model,
    logits_from_output,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_prompt="What is the capital of France?")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    return parser


def generate_text(model, tokenizer, torch, runtime, prompt: str, max_new_tokens: int, temperature: float) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    generated: list[int] = []

    print(f"Input: {prompt}")
    print(f"Initial tokens: {len(input_ids)}")

    for step in range(max_new_tokens):
        seq = torch.tensor([input_ids + generated], dtype=torch.long).to(runtime.tensor_device)

        with torch.no_grad():
            output = model(seq)
        logits = logits_from_output(output)
        next_token_logits = logits[0, -1]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()

        generated.append(next_token)
        token_str = tokenizer.decode([next_token])
        print(f"  step {step + 1}: token={next_token} {token_str!r}")

        if next_token == tokenizer.eos_token_id:
            print(f"[EOS at step {step + 1}]")
            break

    return tokenizer.decode(input_ids + generated)


def main() -> None:
    args = build_parser().parse_args()
    torch = import_torch()
    runtime = resolve_device(torch, args.device)

    model = load_model(args, runtime, torch)
    tokenizer = model.tokenizer
    print(f"Loaded config: {model.config._name_or_path}")

    result = generate_text(
        model,
        tokenizer,
        torch,
        runtime,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
    )
    print(f"Result: {result}")
    print("SUCCESS: AirLLM manual generation completed.")


if __name__ == "__main__":
    main()
