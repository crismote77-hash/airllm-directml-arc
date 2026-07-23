# Agent Notes

## Project State

This checkout is a local AirLLM fork focused on portable runtime smoke tests.
Keep the upstream package structure intact unless a requested change explicitly
requires package internals.

Local scripts:

* `test_arc.py`: forward-pass smoke test.
* `test_v3_arc.py`: `generate()` smoke test with forward fallback.
* `test_arc_gen.py`: manual token-by-token generation smoke test.
* `test_qwen25vl.py`: text-only Qwen2.5-VL smoke/diagnostic script.
* `runtime_utils.py`: shared runtime detection and model-loading helpers.

## Runtime Policy

Use the local `.venv` for validation. Do not commit `.venv`, `__pycache__`, or
generated egg-info artifacts.

Known local environment as of 2026-07-23:

* Python 3.12 `.venv`.
* PyTorch `2.13.0+cpu`.
* Transformers `5.12.1`.
* Intel UHD Graphics 620 is not exposed as a PyTorch XPU device.

For this machine, treat AirLLM as CPU-only. Do not pursue OpenVINO integration
unless the user explicitly reopens that path. OpenVINO was considered and then
stopped by user instruction.

## Device Handling

Do not hardcode:

* Windows model paths such as `D:\...`.
* Intel Arc A770-specific text.
* DirectML-only imports at module import time.
* `.cuda()` in portable scripts.

Use the shared `runtime_utils.py` helpers for CLI arguments, device selection,
dtype selection, and local package import setup.

Supported script device values:

* `auto`
* `cpu`
* `cuda[:N]`
* `xpu[:N]`
* `mps`
* `directml` / `dml[:N]`
* `privateuseone[:N]`

## Validation

Before committing script or docs changes, run:

```bash
.venv/bin/python -m py_compile runtime_utils.py test_arc.py test_v3_arc.py test_arc_gen.py test_qwen25vl.py air_llm/airllm/device_utils.py
.venv/bin/python -m pip check
.venv/bin/python -c "import torch, transformers, airllm; print(torch.__version__, transformers.__version__)"
.venv/bin/python test_arc.py --device cpu --model hf-internal-testing/tiny-random-LlamaForCausalLM --max-new-tokens 1 --layer-shards-saving-path /tmp/airllm-smoke-shards
git diff --check
```

For CPU benchmark context, read `BENCHMARKS.md`.
