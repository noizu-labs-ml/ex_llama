# NIF Interface

## Overview

All NIF functions are defined in `lib/ex_llama/nif.ex` as stubs that raise `:nif_not_loaded`. The native library is loaded via Rustler's `use Rustler` macro. All NIFs are scheduled on `DirtyCpu` to avoid blocking BEAM schedulers.

## Elixir Binding (`lib/ex_llama/nif.ex`)

```elixir
use Rustler,
    otp_app: :ex_llama,
    crate: :erlang_llama_cpp_nif,
    mode: (if Mix.env() == :prod, do: :release, else: :debug)
```

## Model NIFs

| Function | Args | Returns | Purpose |
|----------|------|---------|---------|
| `__model_nif_load_from_file__` | path, model_options | `{:ok, %Model{}}` | Load GGUF model |
| `__model_nif_detokenize__` | model, token_i32 | `{:ok, [byte]}` | Token → bytes |
| `__model_nif_token_to_byte_piece__` | model, token_i32 | `{:ok, [byte]}` | Token → byte piece |
| `__model_nif_token_to_piece__` | model, token_i32 | `{:ok, string}` | Token → string |
| `__model_nif_decode_tokens__` | model, [i32] | `{:ok, string}` | Token list → string |
| `__model_nif_create_session__` | model, session_opts | `{:ok, %Session{}}` | Create inference session |
| `__model_nif_embeddings__` | model, input, embed_opts | `{:ok, [[float]]}` | Generate embeddings |
| `__model_nif_bos__` | model | `{:ok, i32}` | Beginning-of-sequence token |
| `__model_nif_eos__` | model | `{:ok, i32}` | End-of-sequence token |
| `__model_nif_nl__` | model | `{:ok, i32}` | Newline token |
| `__model_nif_infill_prefix__` | model | `{:ok, i32}` | Infill prefix token |
| `__model_nif_infill_middle__` | model | `{:ok, i32}` | Infill middle token |
| `__model_nif_infill_suffix__` | model | `{:ok, i32}` | Infill suffix token |
| `__model_nif_eot__` | model | `{:ok, i32}` | End-of-turn token |
| `__model_nif_vocabulary_size__` | model | `{:ok, int}` | Vocab size |
| `__model_nif_embed_len__` | model | `{:ok, int}` | Embedding dimension |
| `__model_nif_train_len__` | model | `{:ok, int}` | Training context length |

## Session NIFs

| Function | Args | Returns | Purpose |
|----------|------|---------|---------|
| `__session_nif_default_session_options__` | — | `{:ok, %SessionOptions{}}` | Default params |
| `__session_nif_advance_context__` | resource, string | `{:ok, "OK"}` | Add text to context |
| `__session_nif_advance_context_with_tokens__` | resource, [i32] | `{:ok, "OK"}` | Add tokens to context |
| `__session_nif_completion__` | resource, max_tokens, stop_regex | `{:ok, %Completion{}}` | Sync completion |
| `__session_nif_start_completing_with__` | pid, resource, max_tokens | `{:ok, "OK"}` | Streaming completion (sends to PID) |
| `__session_nif_model__` | resource | `{:ok, %Model{}}` | Get session's model |
| `__session_nif_params__` | resource | `{:ok, %SessionOptions{}}` | Get session params |
| `__session_nif_context_size__` | resource | `{:ok, int}` | Current context token count |
| `__session_nif_context__` | resource | `{:ok, [i32]}` | Get context tokens |
| `__session_nif_truncate_context__` | resource, n_tokens | `{:ok, "OK"}` | Truncate context |
| `__session_nif_set_context_to_tokens__` | resource, [i32] | `{:ok, "OK"}` | Replace context with tokens |
| `__session_nif_set_context__` | resource, string | `{:ok, "OK"}` | Replace context with text |
| `__session_deep_copy__` | resource | `{:ok, %Session{}}` | Deep copy session state |

## Resource Types

### ExLLamaModelRef
- Wraps: `LlamaModel` (from llama_cpp crate)
- Thread safety: `unsafe impl Send + Sync` (llama_cpp model is thread-safe for reads)
- Elixir struct: `%ExLLama.Model{resource, name, eos, bos}`

### ExLLamaSessionRef
- Wraps: `Mutex<LlamaSession>`
- Thread safety: Mutex provides mutual exclusion; `unsafe impl Send + Sync`
- Elixir struct: `%ExLLama.Session{seed, model_name, resource}`

## Struct Mappings (Elixir ↔ Rust)

| Elixir | Rust | llama_cpp |
|--------|------|-----------|
| `%ExLLama.ModelOptions{}` | `ModelOptions` | `LlamaParams` |
| `%ExLLama.SessionOptions{}` | `ExLLamaSessionOptions` | `SessionParams` |
| `%ExLLama.EmbeddingOptions{}` | `ExLLamaEmbeddingOptions` | `EmbeddingsParams` |
| `%ExLLama.Completion{}` | `ExLLamaCompletion` | — (constructed from output) |
