# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExLLama is an Elixir NIF wrapper for llama_cpp that enables loading and running GGUF format Large Language Models. It provides both direct text completion and chat-based interfaces with support for multiple chat templates.

## Common Development Commands

### Build and Dependencies
- `mix deps.get` - Fetch dependencies
- `mix compile` - Compile Elixir and Rust NIF code

### Testing
- `mix test` - Run the test suite
- `mix test test/ex_llama_test.exs:LINE_NUMBER` - Run a specific test

### Documentation
- `mix docs` - Generate ExDoc documentation

### Code Quality
- `mix dialyzer` - Run static analysis (first run will build PLT)

### Model Setup
Download test model: `cd priv/models/local_llama/tiny_llama && bash init.sh`

## Architecture

### Two-Layer Design
1. **Elixir Layer** (`/lib/`)
   - High-level API in `ex_llama.ex`
   - Chat templates in `/lib/ex_llama/chat_template/`
   - Session management via `session.ex` and `model.ex`

2. **Rust NIF Layer** (`/native/erlang_llama_cpp_nif/`)
   - Interfaces with llama_cpp crate
   - Handles model loading and inference
   - Resource management for models and sessions

### Key Modules
- `ExLLama` - Main API module with convenience functions
- `ExLLama.Model` - Model loading and management
- `ExLLama.Session` - Inference session handling
- `ExLLama.ChatTemplate` - Template system for different chat formats
- `ExLLama.Nif` - Low-level NIF bindings

### Chat Template System
Supports multiple formats via behavior-based implementations:
- ChatML, Llama2, Llama3Instruct, Mistral, Zephyr, Alpaca, Vicuna
- AmberChat, ChatQA, FalconInstruct, GemmaInstruct, GraniteInstruct
- OpenChat, Phi3, Phi3Small, QwenInstruct, Saiga, SolarInstruct
- Template selection via `meta[:template]` option
- Default: Zephyr template

### Integration Points
- Uses GenAI Core v0.2 structures for standardized AI data types
- Returns `GenAI.ChatCompletion` structs from chat completion
- Optional dependencies: Finch, UUID libraries

## Testing Approach
Tests use TinyLlama model stored in `/priv/models/`. All tests use deterministic seeds for reproducible results. Test patterns include:
- Model loading
- Session creation
- Context advancement
- Synchronous and streaming completion
- Chat completion with multiple choices

## Development Notes
- Rust compilation mode: debug for dev/test, release for prod
- PLT files stored in `priv/plts/` for dialyzer
- Model files excluded from hex package
- Uses rustler v0.32.1 for Elixir-Rust interop

## Known Issues
- Fatal runtime error "Rust cannot catch foreign exceptions" occurs when running the full test suite. This is due to C++ exceptions from llama.cpp that can't be caught by the Rust FFI layer. The chat template tests run successfully in isolation (`mix test test/chat_template_test.exs`).