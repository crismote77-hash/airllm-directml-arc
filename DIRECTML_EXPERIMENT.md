# Intel Arc DirectML experiment report

Status: **technically successful; not usable for interactive large-model generation**

Date: 2026-07-23

Repository: <https://github.com/crismote77-hash/airllm-directml-arc>

## Objective

Determine whether AirLLM 3.0.1 can preserve its layer-by-layer streaming model while
executing Hugging Face causal language models on an Intel Arc A770 through DirectML.

The experiment was a feasibility test, not a performance claim. It did not attempt to
turn GGUF/Ollama models into AirLLM checkpoints or validate vision-language models.

## Validated environment

| Component | Value |
|---|---|
| OS | Windows |
| GPU | Intel Arc A770, 16 GB |
| PyTorch | `2.4.1+cpu` |
| torch-directml | `0.2.5.dev240914` |
| Transformers | `5.12.1` |
| PyTorch device | `privateuseone:0` |
| Validated dtype | `torch.float32` |

The installed DirectML build is compiled against PyTorch 2.4.1. A newer PyTorch test
broke `torch_directml`; Transformers 5.14.1 also failed against PyTorch 2.4.1 because it
expected a newer `DTensor` API. Transformers 5.12.1 was the compatible point that kept
DirectML operational while recognizing newer configuration types.

Float64 failed on DirectML. FP16/BF16 was not validated. AirLLM 4/8-bit compression is
backed by CUDA-oriented `bitsandbytes` and is deliberately rejected on DirectML.

## DirectML changes validated

- Backend selection for CUDA, DirectML/privateuseone, CPU, and MPS.
- Local checkpoint and single-file checkpoint discovery.
- Device-aware cache cleanup and pin-memory decisions.
- Early rejection of CUDA-only compression on DirectML.
- Standard `AutoModel.generate()` on a DirectML tensor.

## Model results

| Model | Architecture | Outcome |
|---|---|---|
| TinyLlama 1.1B | causal LM | PASS: forward and generation |
| Qwen2.5-1.5B-Instruct | `Qwen2ForCausalLM` | PASS: correct Paris answer |
| allura-forge/Llama-3.3-8B-Instruct | `LlamaForCausalLM` | PASS, but not usable interactively |
| Qwen2.5-VL-3B | vision-language | UNSUPPORTED by causal-LM runner |
| Qwen3.5-4B | nested language/vision/MTP | UNSUPPORTED by current splitter |

### TinyLlama

- 22 layers.
- Forward logits shape: `[1, 8, 32000]`.
- Output tensor on `privateuseone:0`.
- Observed layer processing around 6.6-8.3 layers/s.
- `generate()` completed and produced Paris for the capital-of-France prompt.

### Qwen2.5-1.5B

- Local checkpoint size: approximately 3.09 GB.
- 30 split artifacts.
- Initial split took approximately 17 seconds.
- Generated: `What is the capital of France? The capital of France is Paris. It is located in the northwestern part`.

### Llama 3.3 8B

This was the largest completed validation and used a community checkpoint recovered
from Meta's Llama API fine-tuning output, not an official Meta Hugging Face repository:

- Repository: `allura-forge/Llama-3.3-8B-Instruct`.
- Revision: `df95224cf87c32d9f4958dd284a07ded620aa4fc`.
- Checkpoint size: approximately 16.08 GB.
- Split size: approximately 16.06 GB.
- Architecture: `LlamaForCausalLM`, 32 transformer layers.
- Split artifacts: 35/35.
- Initial split/load: 245.52 seconds.
- Prompt: `The capital of France is`.
- Maximum new tokens: 4, deterministic.
- Generation time: 181.86 seconds.
- Throughput: 0.022 tokens/s, approximately 45.47 seconds/token.
- Output: `The capital of France is Paris, and it`.
- Process exit code: 0.

The run proves execution correctness on DirectML. It does not establish acceptable
interactive latency.

## Why the 8B model is so slow

AirLLM reduces resident VRAM by loading one layer at a time. Autoregressive generation
still needs every layer for every new token. On this backend each token therefore causes
repeated disk-to-RAM loading, BF16-to-FP32 conversion, RAM-to-GPU transfer, operator
dispatch, and eviction across all layers.

The 16 GB BF16 checkpoint represents roughly 32 GB when executed in FP32. The Arc GPU
spends much of the 45-second token latency waiting for I/O, conversion, PCIe transfers,
and synchronization rather than sustained matrix compute. KV caching avoids recomputing
prior token states but cannot eliminate per-token layer streaming.

AirLLM's value proposition is capacity: it can make an otherwise unloadable model run.
It does not make that model fast. Increasing model size would worsen this latency, so
32B/72B follow-up downloads are not justified for interactive use on this stack.

For practical 8B inference on the A770, use a quantized GGUF model through llama.cpp's
SYCL backend so the model remains resident in VRAM. That is a different runtime and
format from AirLLM.

## Unsupported paths and paused work

- Qwen2.5-VL requires a vision-language model class, not the generic causal-LM path.
- Qwen3.5 was recognized by Transformers 5.12.1, but the AirLLM splitter failed with
  `ValueError: invalid literal for int() with base 10: 'layers'` because its keys include
  nested language, vision, and MTP groups. Supporting it requires a dedicated adapter.
- Qwen3.5 implementation is intentionally paused and is not part of this closeout.

## Reproducibility and cleanup

`test_llama33_8b.py` is the preserved smoke script. It requires
`AIRLLM_MODEL_PATH` to point to an already-downloaded local checkpoint and never
downloads a model implicitly. Other root diagnostics are likewise local-checkpoint-only.

After collecting the evidence above, all project-local checkpoints, Hugging Face local
caches under `models/`, and generated `splitted_model` directories were deleted. No
model files are distributed by this repository. `models/`, `workspace/`, safetensors,
GGUF, and PyTorch weight shards are excluded by `.gitignore`.

## Final conclusion

AirLLM 3.0.1 can be adapted to execute standard Hugging Face causal LMs on an Intel Arc
A770 through DirectML in FP32. TinyLlama, Qwen2.5-1.5B, and the community Llama 3.3 8B
completed successfully. The 8B result also establishes the practical limit of this
approach: layer streaming at approximately 45 seconds per token is not usable for an
interactive application. The experiment is closed as a feasibility success and a
performance no-go.