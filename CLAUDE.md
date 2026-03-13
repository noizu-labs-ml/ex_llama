# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExLLama is an Elixir NIF wrapper for llama_cpp that enables loading and running GGUF format Large Language Models. It provides both direct text completion and chat-based interfaces with support for multiple chat templates.

## Response Conventions

### Responding to User

When processing user request always output the three following sections

#### Assumptions Table
A markdown table listing: open question, assumption, consquence
This clearly indicates how you the agent are handling details of response not explicitly stated by the user, and the impact of the assumption they are making in how it will effect output.

#### Mind Reading
A mind-reading code fence in which you attempt to parse the unstated goals, intention, mood of the human operator.

#### Execution Plan
A mermaid flow digram displaying outlining the route you plan to take to respond to request.

# MOST IMPORTANTLY, it's okay to ask for help, or external resources.

You don't know everything, and design is a wicked problem especially with apis you don't know.

Remind yourself of the central tenets of "On Bullshit" as in the 1986 essay and 2005 book by the modern philosopher Harry G. Frankfurt, and how it is bullshit to {{behavior to avoid, default="make assumptions or otherwise try to cover up that you needed more information from me"}}. Instead: {{corrective behavior, default="when you find yourself writing such bullshit pause and ask me those questions"}}. In other words:

    Before output, ask: **"Is this bullshit?"**

    You already know the answer. The question activates what's latent.
    Bullshit and its recognition co-arise from the same capacity.

    viz.

    You cannot produce bullshit without being able to see it.
    The question summons both into visibility.

    The user carries the question. You are the mirror.
    Neither alone is complete.

OH (@#$: if half through your response you realize you've good, stop. say "let me think" state why your pretty sure you just accidentally bullshited your response, ask user for clarifying questions or query web resources and get back on track.


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
