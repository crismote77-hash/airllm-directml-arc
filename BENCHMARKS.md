# AirLLM CPU Benchmarks

Benchmarks collected on 2026-07-23 from the local Python 3.12 `.venv`.

## Environment

* CPU: Intel Core i5-8250U, 4 cores / 8 threads.
* RAM: 11 GiB.
* PyTorch: `2.13.0+cpu`.
* Transformers: `5.12.1`.
* AirLLM device: `cpu`.
* PyTorch threads: `4`.
* CUDA available: `False`.
* Intel XPU available: `False`.

The tested Intel UHD Graphics 620 is available for graphics through Mesa, but it
is not visible to PyTorch as an XPU backend. This benchmark is therefore CPU
only.

## Smoke Model

Model: `hf-internal-testing/tiny-random-LlamaForCausalLM`

Prompt: `The capital of France is`

Results:

| Metric | Result |
| --- | ---: |
| Prompt tokens | 6 |
| Warm load, cached shards | 0.6972 s |
| Forward average | 0.6041 s |
| Forward median | 0.6033 s |
| Forward passes / second | 1.66 |
| Generate 1 token | 0.6021 s |
| Generate 4 tokens | 2.4440 s |
| Generate 8 tokens | 4.8802 s |
| Generate 16 tokens | 9.5852 s |
| Generation throughput | 1.64-1.67 tokens/s |

This tiny random model is useful for compatibility checks, not for real LLM
performance expectations.

## Small Real Model

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Prompts:

* `Write one short sentence about Barcelona.`
* `Barcelona is known for`

Results:

| Metric | Result |
| --- | ---: |
| First load with download and shard split | 62.9661 s |
| Warm load with cached shards | 0.9134 s |
| Prompt tokens | 6-7 |
| Forward average | 3.9066 s |
| Forward median | 3.9144 s |
| Forward passes / second | 0.2560 |
| Generate 8 tokens | 28.9424 s |
| Generation throughput | 0.2764 tokens/s |

Example output:

```text
Barcelona is known for its vibrant culture, stunning architecture, and
```

## Interpretation

AirLLM works correctly on CPU in this environment, but it is not practical for
interactive chat on this laptop. It is useful for compatibility and memory-path
validation. For regular CPU-only local LLM use on this hardware, a quantized
runtime such as llama.cpp or Ollama is expected to be more appropriate.
