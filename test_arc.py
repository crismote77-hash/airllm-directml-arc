"""System-agnostic AirLLM forward-pass smoke test."""

from __future__ import annotations

import argparse

from runtime_utils import (
    add_common_args,
    import_torch,
    load_model,
    logits_from_output,
    move_input_ids,
    resolve_device,
    tokenize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_prompt="What is the capital of France?")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch = import_torch()
    runtime = resolve_device(torch, args.device)

    model = load_model(args, runtime, torch)
    tokenizer = model.tokenizer

    print(f"Loaded config: {model.config._name_or_path}")
    print(f"Prompt: {args.prompt}")

    input_tokens = tokenize(tokenizer, args.prompt, args.max_length)
    input_ids = move_input_ids(input_tokens["input_ids"], runtime)
    print(f"Input tokens shape: {tuple(input_ids.shape)}")

    with torch.no_grad():
        output = model(input_ids)

    logits = logits_from_output(output)
    top_token = logits[0, -1].argmax().item()

    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Output device: {logits.device}")
    print(f"Top next token: {top_token} -> {tokenizer.decode([top_token])!r}")
    print("SUCCESS: AirLLM forward pass completed.")


if __name__ == "__main__":
    main()
