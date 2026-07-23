"""System-agnostic AirLLM text-only smoke test for Qwen2.5-VL models."""

from __future__ import annotations

import argparse
import os

from runtime_utils import (
    add_common_args,
    decode_generation,
    ensure_local_airllm_importable,
    import_torch,
    load_model,
    move_input_ids,
    resolve_device,
    tokenize,
)


DEFAULT_QWEN25VL_MODEL = os.getenv("AIRLLM_QWEN25VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        default_model=DEFAULT_QWEN25VL_MODEL,
        default_prompt="What is the capital of France?",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only load and print the Hugging Face config.",
    )
    return parser


def diagnose_config(model_name_or_path: str, hf_token: str | None) -> None:
    ensure_local_airllm_importable()
    from transformers import AutoConfig  # noqa: PLC0415

    token_kwargs = {"token": hf_token} if hf_token else {}
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, **token_kwargs)
    print(f"Architectures: {getattr(config, 'architectures', None)}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Vision config: {getattr(config, 'vision_config', None)}")


def main() -> None:
    args = build_parser().parse_args()

    if args.diagnose_only:
        diagnose_config(args.model, args.hf_token)
        return

    torch = import_torch()
    runtime = resolve_device(torch, args.device)

    try:
        model = load_model(args, runtime, torch)
        tokenizer = model.tokenizer

        print(f"Loaded config: {model.config._name_or_path}")
        print(f"Architectures: {getattr(model.config, 'architectures', None)}")
        print(f"Model type: {getattr(model.config, 'model_type', 'unknown')}")
        print(f"Prompt: {args.prompt}")

        input_tokens = tokenize(tokenizer, args.prompt, args.max_length)
        input_ids = move_input_ids(input_tokens["input_ids"], runtime)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        print(f"Result: {decode_generation(tokenizer, output)}")
        print("SUCCESS: Qwen2.5-VL text-only AirLLM generation completed.")
    except Exception as exc:  # noqa: BLE001 - script should diagnose broad runtime failures.
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("--- Config diagnosis ---")
        diagnose_config(args.model, args.hf_token)
        raise


if __name__ == "__main__":
    main()
