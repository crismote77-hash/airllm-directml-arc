"""Shared helpers for portable AirLLM smoke-test scripts."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SMOKE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@dataclass(frozen=True)
class RuntimeDevice:
    """The device string AirLLM uses and the object tensors should move to."""

    kind: str
    model_device: str
    tensor_device: Any
    description: str


def ensure_local_airllm_importable() -> None:
    """Prefer the checkout's package when scripts are run from the repo root."""

    package_root = Path(__file__).resolve().parent / "air_llm"
    if package_root.exists():
        package_root_s = str(package_root)
        if package_root_s not in sys.path:
            sys.path.insert(0, package_root_s)


def import_torch():
    """Import torch with a clear CLI-facing error if the runtime is missing."""

    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this Python environment.\n"
            "Create/use a compatible environment, then install this checkout, for example:\n"
            "  python -m pip install -e ./air_llm\n"
            "For Windows DirectML also install the extra/runtime package:\n"
            "  python -m pip install torch-directml"
        ) from exc
    return torch


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_prompt: str,
    default_model: str | None = None,
) -> None:
    parser.add_argument(
        "--model",
        default=default_model or os.getenv("AIRLLM_MODEL", DEFAULT_SMOKE_MODEL),
        help=(
            "Hugging Face repo id or local model directory. "
            "Can also be set with AIRLLM_MODEL."
        ),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("AIRLLM_DEVICE", "auto"),
        help=(
            "Runtime device: auto, cuda[:N], xpu[:N], mps, directml/dml[:N], "
            "privateuseone[:N], or cpu. Can also be set with AIRLLM_DEVICE."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("AIRLLM_DTYPE", "auto"),
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Runtime dtype. Defaults to a conservative per-device choice.",
    )
    parser.add_argument("--prompt", default=default_prompt, help="Prompt text.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Tokens to generate.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token.")
    parser.add_argument(
        "--layer-shards-saving-path",
        default=os.getenv("AIRLLM_LAYER_SHARDS_PATH"),
        help="Optional directory for AirLLM split layer shards.",
    )
    parser.add_argument(
        "--compression",
        choices=("4bit", "8bit"),
        default=os.getenv("AIRLLM_COMPRESSION"),
        help="Optional CUDA-only AirLLM compression.",
    )
    parser.add_argument(
        "--no-prefetching",
        action="store_true",
        help="Disable AirLLM layer prefetching.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Let AirLLM delete original downloaded weights after splitting.",
    )


def resolve_device(torch, requested: str) -> RuntimeDevice:
    requested = (requested or "auto").strip().lower()

    if requested == "auto":
        detected = _auto_device(torch)
        if detected is not None:
            return detected
        return _cpu_device(torch)

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("Requested CUDA, but torch.cuda.is_available() is false.")
        return RuntimeDevice("cuda", requested, torch.device(requested), f"NVIDIA CUDA ({requested})")

    if requested.startswith("xpu"):
        if not _xpu_available(torch):
            raise SystemExit("Requested Intel XPU, but torch.xpu.is_available() is false.")
        return RuntimeDevice("xpu", requested, torch.device(requested), f"Intel XPU ({requested})")

    if requested == "mps":
        if not _mps_available(torch):
            raise SystemExit("Requested Apple MPS, but torch.backends.mps.is_available() is false.")
        return RuntimeDevice("mps", "mps", torch.device("mps"), "Apple Metal MPS")

    if requested in {"directml", "dml"} or requested.startswith(("directml:", "dml:", "privateuseone")):
        return _directml_device(torch, requested)

    if requested == "cpu":
        return _cpu_device(torch)

    raise SystemExit(f"Unsupported --device value: {requested}")


def parse_dtype(torch, dtype_name: str, runtime: RuntimeDevice):
    dtype_name = (dtype_name or "auto").lower()
    if dtype_name == "auto":
        if runtime.kind == "cuda":
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_name)


def load_model(args: argparse.Namespace, runtime: RuntimeDevice, torch):
    ensure_local_airllm_importable()
    from airllm import AutoModel  # noqa: PLC0415

    dtype = parse_dtype(torch, args.dtype, runtime)
    if args.layer_shards_saving_path:
        shard_path = Path(args.layer_shards_saving_path).expanduser()
        shard_path.mkdir(parents=True, exist_ok=True)
        args.layer_shards_saving_path = str(shard_path)

    print(f"Runtime: {runtime.description}")
    print(f"AirLLM device: {runtime.model_device}")
    print(f"Tensor device: {runtime.tensor_device}")
    print(f"dtype: {dtype}")
    print(f"Model: {args.model}")

    return AutoModel.from_pretrained(
        args.model,
        device=runtime.model_device,
        dtype=dtype,
        hf_token=args.hf_token,
        layer_shards_saving_path=args.layer_shards_saving_path,
        compression=args.compression,
        prefetching=not args.no_prefetching,
        delete_original=args.delete_original,
    )


def tokenize(tokenizer, prompt: str, max_length: int):
    return tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def move_input_ids(input_ids, runtime: RuntimeDevice):
    return input_ids.to(runtime.tensor_device)


def logits_from_output(output):
    return output.logits if hasattr(output, "logits") else output[0]


def decode_generation(tokenizer, generation_output) -> str:
    if isinstance(generation_output, str):
        return generation_output

    sequences = getattr(generation_output, "sequences", generation_output)
    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    return tokenizer.decode(first, skip_special_tokens=False)


def _auto_device(torch) -> RuntimeDevice | None:
    if torch.cuda.is_available():
        return RuntimeDevice("cuda", "cuda:0", torch.device("cuda:0"), "NVIDIA CUDA (cuda:0)")

    if _xpu_available(torch):
        return RuntimeDevice("xpu", "xpu:0", torch.device("xpu:0"), "Intel XPU (xpu:0)")

    if platform.system().lower() == "darwin" and _mps_available(torch):
        return RuntimeDevice("mps", "mps", torch.device("mps"), "Apple Metal MPS")

    if _directml_available():
        return _directml_device(torch, "directml")

    return None


def _cpu_device(torch) -> RuntimeDevice:
    return RuntimeDevice("cpu", "cpu", torch.device("cpu"), "CPU")


def _xpu_available(torch) -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not hasattr(xpu, "is_available"):
        return False
    try:
        return bool(xpu.is_available())
    except Exception:
        return False


def _mps_available(torch) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def _directml_available() -> bool:
    return importlib.util.find_spec("torch_directml") is not None


def _directml_device(torch, requested: str) -> RuntimeDevice:
    if not _directml_available():
        raise SystemExit("Requested DirectML, but torch-directml is not installed.")

    import torch_directml  # noqa: PLC0415

    index = 0
    if ":" in requested:
        try:
            index = int(requested.rsplit(":", 1)[1])
        except ValueError as exc:
            raise SystemExit(f"Invalid DirectML device index in --device={requested}") from exc

    tensor_device = torch_directml.device(index)
    model_device = str(tensor_device)
    if not model_device.startswith("privateuseone"):
        model_device = f"privateuseone:{index}"

    return RuntimeDevice(
        "directml",
        model_device,
        tensor_device,
        f"DirectML ({model_device})",
    )
