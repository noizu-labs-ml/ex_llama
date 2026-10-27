# NIF Interface

## Overview

The native library is a hand-written C/C++ NIF (`c_src/ex_llama_nif.cpp`)
registered as `ERL_NIF_INIT(Elixir.ExLLama.Nif, …)`. `lib/ex_llama/nif.ex`
declares every NIF as a stub raising `:nif_not_loaded`, and `@on_load :load_nif`
loads `priv/ex_llama_nif`. Heavy calls carry `ERL_NIF_DIRTY_JOB_CPU_BOUND`.

The Elixir module also carries **bridge functions** (`__model_nif_*__`,
`__session_nif_*__`) that translate the historical Rust-era API onto the C NIF
surface (option filtering, struct wrapping, process-dictionary stashing).

## Model NIFs

| Function | Arity | Scheduler | Purpose |
|----------|-------|-----------|---------|
| `load_model` | 1, 2 | DirtyCpu | Load GGUF model → model resource |
| `create_context` | 1, 2 | normal | Create inference context from model (opts) |
| `context_deep_copy` | 1 | DirtyCpu | Deep-copy a context (KV state) |
| `tokenize` | 2, 3 | normal | Text → token list (+ optional add_special) |
| `detokenize` | 2 | normal | Token list → text |
| `token_to_piece` | 2 | normal | Single token → string |
| `vocab_bos` / `vocab_eos` / `vocab_eot` / `vocab_nl` | 1 | normal | Special token ids |
| `vocab_fim_pre` / `vocab_fim_mid` / `vocab_fim_suf` | 1 | normal | Fill-in-middle token ids |
| `model_info` | 1 | normal | Map: n_vocab, n_embd, n_embd_out, n_ctx_train, desc, … |

## Session / Inference NIFs

| Function | Arity | Scheduler | Purpose |
|----------|-------|-----------|---------|
| `completion` | 2, 3 | DirtyCpu | Synchronous completion (ctx, prompt[, opts]) |
| `streaming_completion` | 3, 4 | normal | Send token messages to a PID |
| `embeddings` | 2 | DirtyCpu | Embeddings for text |
| `chat_apply_template` | 2, 3 | normal | llama.cpp chat-template application |

## Classifier NIFs

| Function | Arity | Scheduler | Purpose |
|----------|-------|-----------|---------|
| `classifier_load_head` | 1 | DirtyCpu | Parse JEVH0001 head file → head resource |
| `classifier_head_info` | 1 | normal | `%{hidden, classes}` |
| `classifier_head_probabilities` | 3 | DirtyCpu | (head, features, count) → distribution |
| `classifier_last_hidden` | 3 | DirtyCpu | (model, prompt, opts) → last-token features |

## Elixir Bridge Functions (`lib/ex_llama/nif.ex`)

The `__model_nif_*__` / `__session_nif_*__` names are preserved for API
compatibility and add behaviour the raw NIF lacks:

- **Option filtering**: forward only non-nil keys so llama.cpp defaults survive.
- **Struct wrapping**: return `%ExLLama.Model{}` / `%ExLLama.Session{}`.
- **Process-dictionary session state**: `{:ex_llama_ctx, ref}` (prompt text),
  `{:ex_llama_seed, ref}`, `{:ex_llama_model, ref}`, `{:ex_llama_params, ref}` —
  this is what makes a session portable across processes.
- **Token-level truncation**: tokenize → take n → detokenize.

## Resource Types

### ModelRes
- Owns `llama_model*`; frees via `llama_model_free` in destructor.
- Holds `classifier_mu` (std::mutex) serialising classifier calls.
- Elixir struct: `%ExLLama.Model{resource, name, eos, bos}`.

### CtxRes
- Owns `llama_context*`; holds a non-owning `llama_model*` for vocab access and a `std::mutex`.
- Elixir struct: `%ExLLama.Session{seed, model_name, resource}`.

### HeadRes
- Owns an `ex_llama::JevHead` (hidden, classes, mu, sd, weights, bias).

## Struct Mappings (Elixir → NIF)

| Elixir struct | Purpose | Native effect |
|---------------|---------|---------------|
| `%ExLLama.ModelOptions{}` | load params | `load_model` opts (load_mode, split_mode, …) |
| `%ExLLama.SessionOptions{}` | context/sampling params | `create_context` opts |
| `%ExLLama.ContextParams{}` | same field set as SessionOptions | `create_context` opts |
| `%ExLLama.EmbeddingOptions{}` | embedding thread counts | embeddings |
| `%ExLLama.Completion{}` | completion result | built from output |
