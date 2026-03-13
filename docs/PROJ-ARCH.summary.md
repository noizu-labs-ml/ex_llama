# Architecture Summary

**ExLLama** — Elixir NIF wrapper for llama.cpp enabling GGUF model loading and inference.

## Layers
- **Elixir API** (`lib/`): `ExLLama` → `Model`, `Session`, `ChatTemplate`, `Nif`
- **Native** (`native/erlang_llama_cpp_nif/`): Rust/Rustler NIF → `llama_cpp` crate v0.3.2 → llama.cpp C++

## Resource Model
- Models: `ResourceArc<ExLLamaModelRef>` wrapping `LlamaModel`
- Sessions: `ResourceArc<ExLLamaSessionRef>` wrapping `Mutex<LlamaSession>`
- All NIFs on `DirtyCpu` schedulers
- Completions deep-copy session to avoid holding mutex during inference

## Key Interfaces
- ~17 model NIFs: load, tokenize/detokenize, embeddings, special tokens, metadata
- ~13 session NIFs: context management, sync/streaming completion, deep copy
- 18 chat template implementations via behaviour polymorphism
- Returns `GenAI.ChatCompletion` structs (via `genai_core` dependency)

## Stack
Elixir 1.19 / OTP 28 → Rustler 0.37 → llama_cpp 0.3.2 → llama.cpp → GGUF models
