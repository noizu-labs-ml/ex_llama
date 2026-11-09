ExLLama: Elixir bindings for llama.cpp (C NIF)
=======

<!-- 〚📁:4f8efb43-74a1-50c0-89b4-26a47cf95dd8〛 README.md :: ExLLama project overview and quick-start -->

Elixir bindings for local LLM inference: a **C NIF** wrapping
[llama.cpp](https://github.com/ggml-org/llama.cpp), pinned at **v0.6.0**, built
from source at compile time. Load GGUF models, run chat and raw completions,
stream tokens as they decode, compute embeddings, and execute native Jev
classifiers — all in-process against the local NIF, no external service.

Part of the GenAI libs family (`genai_core`, `genai`). Inspired by
[llama_cpp_ex](https://github.com/jeregrine/llama_cpp_ex).

## Installation

```elixir
def deps do
  [
    {:ex_llama, "~> 0.4.0"}
  ]
end
```

The NIF is compiled by `elixir_make` during `mix compile`. Building it requires
a C/C++ toolchain (C++17), `cmake`, and `git`. Hex installs do not ship the
llama.cpp sources; the Makefile fetches the pinned `v0.6.0` tag from upstream
automatically on first build. The initial compile builds all of llama.cpp and
takes a while; later builds are incremental. Supported platforms: macOS
(Metal + Accelerate) and Linux (OpenBLAS optional). CI validates with
Elixir 1.19 / OTP 28 / cmake 3.31.

## Chat Completion

`ExLLama.chat_completion/3` renders a message thread through a chat template,
runs generation, and returns an `{:ok, %GenAI.ChatCompletion{}}` struct with
per-choice messages and usage totals.

```elixir
{:ok, llama} = ExLLama.load_model("./priv/models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

thread = [
  %{role: :system, content: "You are a helpful assistant."},
  %{role: :user, content: "Say Hello. And only hello. Example \"Hello\"."},
  %{role: :assistant, content: "Hello"},
  %{role: :user, content: "Say Goodbye."}
]

{:ok, completion} =
  ExLLama.chat_completion(llama, thread,
    seed: 2,
    max_tokens: 64,
    choices: 1,
    # optional — defaults to ExLLama.ChatTemplate.Zephyr
    template: ExLLama.ChatTemplate.LLama2Chat
  )

completion.choices |> Enum.map(& &1.message.content)
#=> ["Goodbye"]
```

Template selection: pass `template: <module>` in the options to pick a handler
explicitly; otherwise the default (`ExLLama.ChatTemplate.Zephyr`) is used.

### Built-in chat template catalog

18 templates ship under `lib/ex_llama/chat_template/`, all implementing the
`ExLLama.ChatTemplate` behaviour (`support_list/0`, `to_context/3`,
`extract_response/3`):

| Module | Family |
|---|---|
| `ExLLama.ChatTemplate.Alpaca` | Alpaca instruct |
| `ExLLama.ChatTemplate.AmberChat` | Amber Chat |
| `ExLLama.ChatTemplate.ChatML` | ChatML (`<|im_start|>` — Zephyr, Hermes, ...) |
| `ExLLama.ChatTemplate.ChatQA` | NVIDIA ChatQA |
| `ExLLama.ChatTemplate.FalconInstruct` | Falcon Instruct |
| `ExLLama.ChatTemplate.GemmaInstruct` | Gemma Instruct |
| `ExLLama.ChatTemplate.GraniteInstruct` | IBM Granite Instruct |
| `ExLLama.ChatTemplate.LLama2Chat` | Llama 2 Chat |
| `ExLLama.ChatTemplate.Llama3Instruct` | Llama 3 Instruct |
| `ExLLama.ChatTemplate.MistralInstruct` | Mistral Instruct |
| `ExLLama.ChatTemplate.OpenChat` | OpenChat |
| `ExLLama.ChatTemplate.Phi3` | Phi-3 |
| `ExLLama.ChatTemplate.Phi3Small` | Phi-3 Small |
| `ExLLama.ChatTemplate.QwenInstruct` | Qwen Instruct |
| `ExLLama.ChatTemplate.Saiga` | Saiga |
| `ExLLama.ChatTemplate.SolarInstruct` | SOLAR Instruct |
| `ExLLama.ChatTemplate.Vicuna` | Vicuna |
| `ExLLama.ChatTemplate.Zephyr` | Zephyr (default) |

Custom templates: implement the `ExLLama.ChatTemplate` behaviour in your own
module and pass it via `template:`.

## Streaming Completion

`ExLLama.Session.start_completing_with/2` runs a completion against the
session's current context and streams each decoded chunk to an Erlang process
as a message — the default target is the calling process; pass `pid:` to
redirect.

```elixir
{:ok, options} = ExLLama.Session.default_options()
{:ok, session} = ExLLama.create_session(llama, %{options | seed: 2})
ExLLama.advance_context(session, "Say Hello. And only hello.\n")

:ok = ExLLama.Session.start_completing_with(session, %{max_tokens: 512, pid: self()})

collect = fn collect, acc ->
  receive do
    {:ok, chunk} -> collect.(collect, [chunk | acc])
    {:error, reason} -> {:error, reason, Enum.reverse(acc)}
    :fin -> {:ok, acc |> Enum.reverse() |> IO.iodata_to_binary()}
  after
    60_000 -> {:timeout, Enum.reverse(acc)}
  end
end

{:ok, text} = collect.(collect, [])
```

Message protocol: one `{:ok, text_chunk}` message per decoded chunk, `{:error,
reason}` on failure, and a final `:fin` when generation completes. The stream
is delivered via raw process messages (not an Elixir `Stream`) — consume them
in a `receive` loop as above.

## Embeddings

Embedding-capable GGUF models (e.g. `nomic-embed-text`) can produce vectors
via `ExLLama.Model.embeddings/3`. Each call spins up an embeddings-enabled
context internally; `ExLLama.Model.embed_len/1` reports the vector dimension.

```elixir
{:ok, model} = ExLLama.load_model("./nomic-embed-text-v1.5.Q5_K_M.gguf")

{:ok, vectors} =
  ExLLama.Model.embeddings(model, ["hello world", "goodbye"], %ExLLama.EmbeddingOptions{})
# {:ok, [[floats], [floats]]} — one vector per input

{:ok, dim} = ExLLama.Model.embed_len(model)
```

`ExLLama.EmbeddingOptions` carries `n_threads` / `n_threads_batch` for the
embedding pass. Embedding behavior is tuned through the 0.4.0 context options
(`pooling_type`, `attention_type` — see below).

## Raw Sessions (low-level)

For full control, drive a session directly: `advance_context` appends raw text
to the model context, and `completion` generates from it. No chat template is
applied on this path — you own the prompt format.

```elixir
{:ok, options} = ExLLama.Session.default_options()
{:ok, session} = ExLLama.create_session(llama, %{options | seed: 2})

:ok = ExLLama.advance_context(session, "The capital of France is")
{:ok, %{content: text, token_length: n}} = ExLLama.Session.completion(session, 16, "\n")
```

Session utilities: `set_context/2`, `context/1`, `context_size/1`,
`truncate_context/2`, `advance_context_with_tokens/2`, `set_context_to_tokens/2`,
`params/1`, `model/1`, and `deep_copy/1` (fork a session's state, e.g. to
explore multiple continuations). Model-level helpers: `ExLLama.Model.load_from_file/2`,
`vocabulary_size/1`, `embed_len/1`, `train_len/1`, tokenize/detokenize helpers,
and vocabulary special-token accessors (`bos/eos/nl/eot`, infill tokens).

## 0.4.0 Model & Session Options

The 0.4.0 pin refresh (llama.cpp `b8322` → `v0.6.0`) exposed the new upstream
loading and context options. Full delta catalog:
[docs/llama-cpp-feature-audit.md](docs/llama-cpp-feature-audit.md).

Model loading — `ExLLama.ModelOptions`:

- `load_mode` replaces the old `use_mmap`/`use_mlock` booleans:
  `"auto" | "none" | "mmap" | "mlock" | "mmap+mlock" | "direct_io"`
  (default `"auto"`). The deprecated booleans are still accepted and
  translated to a `load_mode`; an explicit `:load_mode` always wins.
- New: `lazy_mode` (`0` off / `1` auto / `2` on — lazy tensor reads, requires
  mmap), `split_mode` (`"none" | "layer" | "row" | "tensor"` GPU split),
  `main_gpu`, `check_tensors`, `use_extra_bufts`, `no_host`, and `load_mtp`
  (loads MTP speculative heads on GLM-5 / Qwen4Exp-style models).
- Fields left `nil` preserve llama.cpp upstream defaults.

Session/context — `ExLLama.SessionOptions` / `ExLLama.ContextParams`:

- `flash_attn_type` (-1 auto / 0 off / 1 on), `pooling_type` (-1..4),
  `attention_type` (-1 unspecified / 0 causal / 1 non-causal),
- KV-cache quantization via `type_k` / `type_v` (ggml type ints — `1` F16,
  `8` Q8_0; both default to F16),
- `n_ubatch`, `n_seq_max`, `rope_scaling_type`, YaRN overrides (`yarn_*`),
  `op_offload`, `kv_unified`, `swa_full`.
- `ExLLama.Nif.model_info/1` now includes `:desc`.

Model support follows the pin: the 24 GGUF architectures added since `b8322`
(GLM-5/GLM5-Next, DeepSeek 4/32/2-OCR, Gemma 4, Mistral 4, Qwen4Exp+MTP,
Eagle 3, ...) and the Q1_0/Q2_0 quant formats load with no API change.

# Native classifier support

`ExLLama.load_classifier/3` loads a Jev-Omni GGUF backbone and its separate FP32
decision head. `ExLLama.classify/3` implements the Core classifier contract for
text choices, booleans and ordinal rubrics. See [native classifier setup and
validation](docs/jev-classifier.md), including the explicit pending native/live
validation prerequisites.

## Docs

- [docs/jev-classifier.md](docs/jev-classifier.md) — native classifier setup.
- [docs/llama-cpp-feature-audit.md](docs/llama-cpp-feature-audit.md) — full
  `b8322` → `v0.6.0` delta catalog behind the 0.4.0 options surface.
- [docs/arch/](docs/arch/) — chat templates, data flow, and the NIF interface.
