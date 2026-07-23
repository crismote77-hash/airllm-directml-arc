# Agent Notes

## Purpose and status

This checkout is an AirLLM 3.0.1 fork for portable runtime smoke tests and the
completed Intel Arc / DirectML feasibility experiment. Keep the upstream package
structure intact unless an authorized change explicitly requires package internals.

The Arc experiment proved that causal Hugging Face models can execute through
`torch-directml`, but layer streaming in FP32 is too slow for practical interactive use
at 8B scale. The validated Llama run loaded/split in 245.52 seconds and produced four
tokens in 181.86 seconds (`0.022` token/s; about 45.47 seconds/token). It used
`allura-forge/Llama-3.3-8B-Instruct` at revision
`df95224cf87c32d9f4958dd284a07ded620aa4fc`.

Treat DirectML AirLLM as technically functional but **not usable** for interactive
large-model serving. Do not describe it as a practical serving solution.

## Validated runtime profiles

### Portable CPU profile

- Python 3.12 local `.venv`.
- PyTorch `2.13.0+cpu`.
- Transformers `5.12.1`.
- Intel Core i5-8250U with Intel UHD Graphics 620.
- CUDA and PyTorch XPU unavailable; use CPU only on that machine.
- AirLLM is technically functional there but not practical for interactive chat.

Do not pursue OpenVINO for the UHD 620 profile unless the user explicitly reopens that
path. The user stopped that investigation.

### Intel Arc DirectML profile

The only validated Windows tuple is:

- `torch==2.4.1` (`2.4.1+cpu` at runtime)
- `torch-directml==0.2.5.dev240914`
- `transformers==5.12.1`
- `privateuseone:0`
- `torch.float32`

Do not upgrade PyTorch without a matching DirectML build and a fresh smoke test. Do not
claim FP16, BF16, 4-bit, or 8-bit DirectML support. AirLLM compression uses
`bitsandbytes` and is CUDA-only in this fork.

## Device handling

Portable scripts must not hardcode Windows model paths, Arc-specific wording,
DirectML-only imports at module import time, or `.cuda()`. Use `runtime_utils.py` for
CLI arguments, device selection, dtype selection, and local-package setup.

Supported script devices are `auto`, `cpu`, `cuda[:N]`, `xpu[:N]`, `mps`,
`directml` / `dml[:N]`, and `privateuseone[:N]`.

Local scripts:

- `test_arc.py`: portable forward-pass smoke test.
- `test_v3_arc.py`: portable `generate()` smoke test with forward fallback.
- `test_arc_gen.py`: portable manual token-by-token generation smoke test.
- `test_qwen25vl.py`: portable text-only Qwen2.5-VL diagnostic.
- `runtime_utils.py`: shared runtime and model-loading helpers.
- `test_llama33_8b.py`: frozen local-checkpoint reproducer for the Arc benchmark; it
  never downloads a model.

## Architecture boundaries

- DirectML evidence exists for TinyLlama, Qwen2.5-1.5B, and a community Llama 3.3 8B
  causal LM.
- Qwen2.5-VL is outside the current `AutoModelForCausalLM` path.
- Qwen3.5 support is paused. Its nested language/vision/MTP paths require a dedicated
  adapter and must not be resumed or claimed as supported without a new authorized task.
- AirLLM accepts Hugging Face safetensors/bin checkpoints, not GGUF/Ollama models.
- Prefer llama.cpp SYCL plus GGUF when interactive performance is the objective on Arc.

## Model and test hygiene

- Never commit checkpoints, shards, tokenizers, Hugging Face caches, or
  `splitted_model` directories.
- `models/`, `workspace/`, `*.safetensors`, `*.gguf`, and `pytorch_model*.bin` are
  ignored.
- Model-dependent smoke tests require an explicitly authorized download and execution
  cost. Do not run them after cleanup merely to satisfy a generic test gate.
- Document exact model repository, revision, environment, prompt, output, exit code,
  and timing. Never invent missing stdout or throughput.

## Root publication whitelist

Only these root paths belong in the clean public branch without a separate scope review:

- `.gitignore`
- `README.md`
- `AGENTS.md`
- `BENCHMARKS.md`
- `DIRECTML_EXPERIMENT.md`
- `requirements.txt`
- `runtime_utils.py`
- `test_arc.py`
- `test_arc_gen.py`
- `test_qwen25vl.py`
- `test_v3_arc.py`
- `test_llama33_8b.py`

Do not use `git add .` or `git add -A`. The worktree contains unrelated untracked
upstream-looking directories. Stage explicit paths, inspect the staged path list, and
scan staged content for secrets and model weights. Push only `main-clean` to the
authorized `origin`, never force-push, and never publish the upstream-derived history
that previously contained notebook credentials.

## Validation

Use the profile-specific Python environment. For syntax/import-only validation:

```bash
.venv/bin/python -m py_compile runtime_utils.py test_arc.py test_v3_arc.py test_arc_gen.py test_qwen25vl.py test_llama33_8b.py air_llm/airllm/device_utils.py
.venv/bin/python -m pip check
.venv/bin/python -c "import torch, transformers, airllm; print(torch.__version__, transformers.__version__)"
git diff --check
```

Run a model smoke only when its download and runtime cost are explicitly authorized.
Before publication, also verify `models/` is absent, no model extension is staged, the
staged-only secret scan passes without printing values, and local HEAD equals the remote
`origin/main-clean` readback after a non-force push.

Read `BENCHMARKS.md` for CPU results and `DIRECTML_EXPERIMENT.md` for Arc results.
