# llama.cpp Feature Audit — b8322 → v0.6.0

Date: 2026-10-07 · Pin refresh for `ex_llama` 0.4.0.

Old pin: [b8322](https://github.com/ggml-org/llama.cpp/tree/b8322) (2026-08 era)
New pin: [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0) (released 2026-10-05, commit `d81235049`)

Method: header diff of `include/llama.h` + `ggml/include/ggml.h` between the two
tags, plus `src/llama-arch.h` architecture registry diff and the v0.6.0 release
notes. Status legend: **exposed** = plumbed through the NIF into the Elixir API
as of this branch; **newly-exposed** = existed upstream before but only now
surfaced; **skipped** = deliberately not surfaced (reason given).

---

## 1. GGUF architectures / quant formats

### 1a. Newly supported architectures (24 added since b8322)

`LLM_ARCH_*` registry grew 124 → 156 entries; the 24 genuinely new model
families:

BAILINGMOE3, CLEF, COHERE2MOE, DEEPSEEK2OCR, DEEPSEEK32, DEEPSEEK4, DFLASH,
DOTS3NOTE, EAGLE3, GEMMA4, GLM5 (GLM-5.3-Flash / GLM5-Next 320B hybrid), HRM,
HY, LAGUNA, MAPLE, MELLUM, MISTRAL4, MUSE, NANBEIGE, POCKETTTS, QWEN3TTS,
QWEN4EXP (with MTP), SPARK2, TALKIE.

**Status: exposed (implicitly).** Architecture support is a loader-level
concern: any GGUF these arches produce now loads through
`ExLLama.load_model/2` with no per-arch flag needed. No Elixir-side action
required. Nothing to expose explicitly.

### 1b. Quant formats

New `ggml_type` entries since b8322: `GGML_TYPE_Q1_0` (=41), `GGML_TYPE_Q2_0`
(=42). (Ternary TQ1_0/TQ2_0 already existed at b8322.)

**Status: exposed (implicitly).** Deserialization is internal to ggml; models
quantized with these formats simply load. The only place quant *types* appear
in our API is KV-cache `type_k`/`type_v` (see §2b), which accept any
`ggml_type` integer.

---

## 2. Model-loading options (`llama_model_params`)

### 2a. Changed since b8322 (compile-breaking)

| Field | b8322 | v0.6.0 | Status |
|---|---|---|---|
| `use_mmap` / `use_mlock` / `use_direct_io` | `bool` | **removed** | **exposed via replacement** |
| `load_mode` | — | `enum llama_load_mode` (-1 auto, 0 none, 1 mmap, 2 mlock, 3 mmap+mlock, 4 direct_io) | **exposed** as `load_mode` (string `"auto"|"none"|"mmap"|"mlock"|"mmap+mlock"|"direct_io"` or integer) in `ExLLama.ModelOptions` |
| `lazy_mode` | — | `enum llama_lazy_mode` (0 off, 1 auto, 2 on; requires mmap) | **exposed** as `lazy_mode` (0/1/2) |
| `load_mtp` | — | `bool` — load MTP layers (GLM-5/Qwen4Exp style speculative heads) | **exposed** as `load_mtp` |

Note: `llama_load_mode_from_str/1` `GGML_ABORT`s on unknown input, so the NIF
validates strings itself and returns `{:error, reason}` instead of crashing the
VM.

### 2b. Pre-existing but previously unexposed

| Field | Meaning | Status |
|---|---|---|
| `split_mode` | none/layer/row/tensor GPU split | **newly-exposed** (string); default left `nil` so upstream default (`layer`) is preserved |
| `main_gpu` | GPU for the whole model under `split_mode: none` | **newly-exposed** |
| `check_tensors` | validate tensor data at load | **newly-exposed** |
| `use_extra_bufts` | extra buffer types / weight repacking | **newly-exposed**; default left `nil` (upstream default is `true`) |
| `no_host` | bypass host buffer for extra buffers | **newly-exposed** |

### 2c. Deliberately skipped

| Field | Why skipped |
|---|---|
| `no_alloc` | metadata-only simulated load — produces a model that cannot infer; misleading behind a generic flag |
| `progress_callback` / `progress_callback_user_data` | C function pointers; would need a NIF-safe message protocol to be useful — follow-up if wanted |
| `kv_overrides` / `tensor_buft_overrides` / `tensor_split` | C arrays of structs/floats; awkward and unsafe to marshal as flat maps without a schema; not requested |
| `devices` | backend device list; macOS/Metal single-device setups don't need it |

---

## 3. Context options (`llama_context_params`)

### 3a. Changed since b8322

Added upstream: `n_rs_seq` (recurrent-state rollback snapshots),
`n_outputs_max`, `n_outputs_max_per_seq`, `ctx_type` (MTP contexts),
`ctx_other` (context sharing), backend sampler chain (`samplers`/`n_samplers`).
All **skipped**: experimental or require cross-resource NIF bookkeeping with
no current consumer. Sampling through `ctx_other`-shared chains changes
lifetime semantics the current resource DTORs don't model.

### 3b. Newly-exposed (pre-existing upstream, now surfaced in `ExLLama.SessionOptions` / `ContextParams`)

| Field | Meaning | Notes |
|---|---|---|
| `flash_attn_type` | -1 auto / 0 off / 1 on | Metal tensor-API FA kernel is a v0.6.0 headline perf feature |
| `pooling_type` | -1 unspecified / 0 none / 1 mean / 2 cls / 3 last / 4 rank | embeddings/reranking control |
| `attention_type` | -1 unspecified / 0 causal / 1 non-causal | embedding models |
| `type_k` / `type_v` | KV-cache ggml type (1 F16, 8 Q8_0, 2 Q4_0, 30 BF16) | KV-cache quantization |
| `n_ubatch` | physical batch size | |
| `n_seq_max` | max sequences | |
| `rope_scaling_type` | 0 none / 1 linear / 2 yarn | was in the struct, never marshalled |
| `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`, `yarn_orig_ctx` | YaRN overrides | were in the struct, never marshalled |
| `op_offload` | offload host tensor ops to device | |
| `kv_unified` | unified KV buffer across sequences | perf knob for multi-seq |
| `swa_full` | full-size SWA cache | affects sliding-window models |

### 3c. Deliberately skipped

| Field | Why skipped |
|---|---|
| `no_perf` | perf timings toggle; no consumer |
| `defrag_thold` | marked `[DEPRECATED]` upstream |
| `abort_callback` / `cb_eval` | C callbacks (same rationale as progress_callback) |
| `n_rs_seq` / `n_outputs_max*` / `ctx_type` / `ctx_other` / backend samplers | see §3a |

---

## 4. Sampling (`llama_sampler_*`)

**No API change since b8322** — the sampler surface (top_k/top_p/min_p/temp/
dist/typical/xtc/top_n_sigma/mirostat/grammar/penalties/dry/logit_bias/infill)
is identical between the pins.

The completion path keeps its existing chain (top_k → top_p → min_p → temp →
dist). The remaining sampler family (**typical_p, penalties, mirostat, XTC,
top-n-sigma, grammar, DRY, logit_bias**) is **skipped in this branch** — none
of it is new upstream, and wiring it means designing option forwarding through
`Session.completion/3` (which currently passes only `max_tokens`/`stop`).
Follow-up candidate, tracked as future work rather than a refresh gap.

---

## 5. Other API deltas noted

- `llama_batch_ext` + `llama_process`: new extended batch API for mixed
  token/embedding inputs and MTP/deepstack state embeddings. **Skipped** — the
  NIF's simple `llama_batch_get_one`/`llama_decode` path is sufficient for the
  current Elixir surface.
- `llama_model_init_from_user`, `llama_model_load_from_splits`: metadata/split
  loading. **Skipped** — no consumer; `load_from_splits` needs a paths list
  API shape decision.
- `model_info/1` result now includes `:desc` (via `llama_model_desc`) —
  backward-compatible addition.
- `llama_vocab_sep/pad/mask` and `llama_model_is_recurrent/is_hybrid/
  is_diffusion`, `n_embd_inp/out`: introspection niceties, **skipped** (no
  consumer; `model_info` covers the common fields).
- New vocab/model helpers deprecated in favor of `llama_vocab_*`/
  `llama_model_*` names — the NIF already used the non-deprecated spellings,
  no action.

---

## 6. Build-system deltas

- CMake layout unchanged for our targets: `libllama.a`, `libggml.a`,
  `libggml-base.a`, `libggml-cpu.a`, `ggml-blas/libggml-blas.a`,
  `ggml-metal/libggml-metal.a` all still produced at the same paths.
- `LLAMA_BUILD_COMMON` now defaults to ON for standalone builds — Makefile
  pins `-DLLAMA_BUILD_COMMON=OFF` (and `-DLLAMA_BUILD_TOOLS=OFF`) to keep the
  NIF build lean.
- `cmake` 3.31.11 (asdf toolchain) builds the pin cleanly on macOS with
  Metal + Accelerate BLAS.
