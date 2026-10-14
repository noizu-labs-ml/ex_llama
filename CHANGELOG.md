Change Log
=====

## 0.4.0
Refresh llama.cpp submodule pin from tag `b8322` to stable release
`v0.6.0` (2026-10-05), and expose the new/previously-unexposed model-loading
and context options in the Elixir API. Full delta catalog:
`docs/llama-cpp-feature-audit.md`.

Model loading (`ExLLama.ModelOptions`):

- `use_mmap`/`use_mlock` are gone — llama.cpp replaced them with a single
  `load_mode` (`"auto"|"none"|"mmap"|"mlock"|"mmap+mlock"|"direct_io"`).
- New: `lazy_mode` (0/1/2), `split_mode` (`"none"|"layer"|"row"|"tensor"`),
  `main_gpu`, `check_tensors`, `use_extra_bufts`, `no_host`, `load_mtp`.
  Fields left `nil` preserve upstream defaults.

Session/context (`ExLLama.SessionOptions` / `ContextParams`):

- New: `flash_attn_type` (-1/0/1), `pooling_type` (-1..4), `attention_type`
  (-1/0/1), KV-cache quantization via `type_k`/`type_v` (ggml type ints,
  e.g. 1 F16 / 8 Q8_0), `n_ubatch`, `n_seq_max`, `rope_scaling_type`,
  YaRN overrides (`yarn_*`), `op_offload`, `kv_unified`, `swa_full`.
- `model_info/1` now includes `:desc`.

Model support follows the pin: 24 new GGUF architectures since b8322
(GLM-5/GLM5-Next, DeepSeek 4/32/2-OCR, Gemma 4, Mistral 4, Qwen4Exp+MTP,
Eagle 3, Clef, ...) and Q1_0/Q2_0 quant formats load with no API change.

Build: `LLAMA_BUILD_COMMON/TOOLS/APP` now pinned OFF (v0.6.0 defaults them
ON for standalone builds and the unified `llama` app fails without the
server lib). Hygiene: built `priv/ex_llama_nif.so` untracked and gitignored.

## 0.3.0
- Merge the `llama.cpp` release line (0.2.x source of truth) into `develop`.
- Pin `genai_core` to `~> 0.3.5` (from `~> 0.3`).

## 0.2.3
Complete the Linux build fix started in 0.2.2:

- Pass -DCMAKE_POSITION_INDEPENDENT_CODE=ON so static archives can be
  linked into the NIF .so without "relocation ... can not be used when
  making a shared object" errors on any Linux arch.
- Add -lopenblas -lgomp -ldl to the Linux link line so sgemm_/dgemm_
  (from libggml-blas.a), OpenMP symbols (from libggml-cpu.a), and
  dlopen (from libggml.a) resolve at both link time and runtime.

Darwin link line unchanged.

## 0.2.2
Fix Linux build: enable GGML_BLAS=ON in the cmake step so libggml-blas.a
is produced, matching what the NIF link line already expects. On Darwin
this flag picks Apple Accelerate (prior default); on Linux it selects
OpenBLAS (requires libopenblas-dev at build time).

## 0.2.0
C NIF rewrite: switch from the outdated Rust wrapper to a C NIF built
against the latest llama.cpp. (Published to Hex as 0.2.0–0.2.1 from the
`llama.cpp` branch; in-tree changelog backfilled here.)

## 0.1.0
Update to use GenAI Core structs
