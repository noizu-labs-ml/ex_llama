# Architecture Summary

**ExLLama** — Elixir C-NIF wrapper around llama.cpp for GGUF model loading and inference,
plus a GenAI-Core classifier provider (Jev-Omni backbone + FP32 decision head).

## Layers
- **Elixir API** (`lib/`): `ExLLama` → `Model`, `Session`, `ChatTemplate`, `Classifier`, `Nif`
- **Native** (`c_src/`): hand-written C/C++ NIF → llama.cpp `v0.6.0` (submodule); `jev_head.h`
  parses the JEVH0001 decision head. No Rust in the tree (rewritten at 0.2.0).

## Resource Model
- Models: NIF resource `ModelRes` owning `llama_model*`
- Sessions: NIF resource `CtxRes` (`llama_context*` + non-owning model + mutex)
- Classifier head: NIF resource `HeadRes` (FP32 mu/sd/weights/bias)
- Inference NIFs on `DirtyCpu`; vocab lookups on normal schedulers
- Session context/seed/params stashed in the calling process dictionary keyed by ctx ref

## Key Interfaces
- ~30 NIFs: model (load, tokenize/detokenize, vocab, model_info), session
  (completion, streaming_completion, embeddings, create_context, deep_copy),
  classifier (load_head, head_info, head_probabilities, last_hidden),
  chat_apply_template
- 18 chat template implementations via behaviour polymorphism (default Zephyr)
- Returns `GenAI.ChatCompletion` / `GenAI.Classifier` structs (via `genai_core`)

## Stack
Elixir 1.19 / OTP 28 → C NIF → llama.cpp v0.6.0 → GGUF models (cmake static build)
