# Project Schema

## Scope

ExLLama has **no relational persistence layer** — no Ecto schemas, no Liquibase
changelogs, no database connections. Its "data" is: in-memory option structs
that parameterise the NIF, two binary file formats it consumes (GGUF models and
JEVH0001 decision heads), the process-dictionary session state, and the request/
response structs of the GenAI Core interfaces. This document is the single source
of truth for those shapes. ERD sections do not apply.

→ *Where each lives: see [PROJ-LAYOUT.md](PROJ-LAYOUT.md)*

## In-Memory Option Structs

Defined under `lib/ex_llama/`. `nil` fields mean "use the llama.cpp default" —
the NIF bridge forwards only non-nil keys.

### ExLLama.ModelOptions (`lib/ex_llama/model_options.ex`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| n_gpu_layers | non_neg_integer | 0 (bridge) | layers offloaded to GPU |
| split_mode | string | nil | `"none"｜"layer"｜"row"｜"tensor"` |
| main_gpu | non_neg_integer | nil | |
| load_mode | string | nil | `"auto"｜"none"｜"mmap"｜"mlock"｜"mmap+mlock"｜"direct_io"` |
| lazy_mode | 0｜1｜2 | nil | |
| vocab_only | boolean | nil | |
| check_tensors | boolean | nil | |
| use_extra_bufts | boolean | nil | |
| no_host | boolean | nil | |
| load_mtp | boolean | nil | |
| use_mmap / use_mlock | boolean｜nil | nil | **deprecated**, resolved into `load_mode` |

### ExLLama.SessionOptions (`lib/ex_llama/session_options.ex`) and ExLLama.ContextParams

Same field set; defaults from `__session_nif_default_session_options__/0`:
`seed 0xFFFFFFFF`, `n_ctx 2048`, `n_batch 512`, `n_threads/-batch -1`,
`rope_scaling_type -1`, `type_k/type_v 1` (F16), `embedding false`,
`offload_kqv true`, yarn `*_factor` nil (inherit model). Also carries
`flash_attn_type`, `pooling_type`, `attention_type`, `n_ubatch`, `n_seq_max`,
`rope_freq_base/scale`, yarn overrides, `op_offload`, `kv_unified`, `swa_full`,
`pooling`. `type_k/type_v` are ggml type ints (1 F16, 8 Q8_0, 2 Q4_0, 30 BF16).

### ExLLama.EmbeddingOptions

`n_threads` (default 1), `n_threads_batch` (default 1).

## Binary File Formats Consumed

### GGUF model files
Standard llama.cpp GGUF (external format; parsed by the pinned llama.cpp
submodule). Supported architectures follow the v0.6.0 pin — see
[llama-cpp-feature-audit.md](llama-cpp-feature-audit.md).

### JEVH0001 decision head (`c_src/jev_head.h`)

Little-endian binary, parsed with strict validation (never a pickle):

```
offset 0    : magic "JEVH0001" (8 bytes)
offset 8    : uint32 hidden   (1..65536)
offset 12   : uint32 classes  (must equal 256)
then floats : mu[hidden], sd[hidden],
              weights[hidden*classes] (row-major), bias[classes]
size        : 16 + (2*hidden + hidden*classes + classes) * 4  (<= 128 MiB)
```

The head computes `(features - mu) / sd` then a linear layer; softmax over the
requested option count gives the distribution.

## Session State Schema (process dictionary)

Keyed by the native context reference (`ctx_ref`); written/read by
`ExLLama.Nif` and `ExLLama.Session`:

| Key | Value | Written by |
|-----|-------|-----------|
| `{:ex_llama_ctx, ref}` | prompt/context text (binary) | advance/set/truncate_context |
| `{:ex_llama_seed, ref}` | integer seed | create_session, ensure_seed |
| `{:ex_llama_model, ref}` | `%ExLLama.Model{}` | create_session |
| `{:ex_llama_params, ref}` | `%ExLLama.SessionOptions{}` | create_session |

## Interface Schemas

### Completion result
`{:ok, %{content: String.t, token_length: integer}}` (sync); streaming emits
token messages then `:fin`.

### Chat template meta (`ExLLama.ChatTemplate`)
`support_list/0`, `to_context(thread, model, meta)`, and
`extract_response([{tokens, text}], model, meta)` returning `GenAI.ChatCompletion`.
Selection: `meta[:template]`, default `ExLLama.ChatTemplate.Zephyr`.

### GenAI classifier contract (`ExLLama.Classifier`)
`classifier_capabilities/0` → `%{types: [:choice, :boolean, :score], modalities: [:text], max_choices: 256, native_multi_question: false}`.
`classify(request, options)` returns `%GenAI.Classifier.Result{}` with per-question
`Answer` (`index`, `choice_id`, `distribution`, `selected_probability`, `value`,
plus `expected_score`/`grade` for scores). Answers require a normalized
distribution (sum ≈ 1, each in [0,1], 2..256 options).

## Migration History

None — the project owns no database schema.
