"""System-agnostic AirLLM generate() smoke test with forward fallback."""

from __future__ import annotations

import argparse

from runtime_utils import (
    add_common_args,
    decode_generation,
    import_torch,
    load_model,
    logits_from_output,
    move_input_ids,
    resolve_device,
    tokenize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_prompt="The capital of France is")
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

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        print(f"Result: {decode_generation(tokenizer, output)}")
        print("SUCCESS: AirLLM generate() completed.")
    except Exception as exc:  # noqa: BLE001 - fallback script should show original failure.
        print(f"generate() failed: {type(exc).__name__}: {exc}")
        with torch.no_grad():
            output = model(input_ids)
        logits = logits_from_output(output)
        print(f"Forward logits shape: {tuple(logits.shape)}")
        print(f"Forward output device: {logits.device}")
        print("SUCCESS: AirLLM forward fallback completed.")


if __name__ == "__main__":
    main()
