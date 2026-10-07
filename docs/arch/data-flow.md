# Data Flow

## Model Loading

```
User: ExLLama.load_model(path, opts)
  → ExLLama.Model.load_from_file(path, opts)
    → ExLLama.Nif.__model_nif_load_from_file__(path, %ModelOptions{})
      → [DirtyCpu] Rust: LlamaModel::load_from_file(path, LlamaParams)
        → Returns ResourceArc<ExLLamaModelRef> wrapping LlamaModel
      → Builds ExLLamaModel{resource, name, eos, bos}
    → Returns %ExLLama.Model{resource: ref, name: path, eos: bytes, bos: bytes}
```

## Session Creation

```
User: ExLLama.create_session(model, opts)
  → ExLLama.Nif.__model_nif_create_session__(model, %SessionOptions{})
    → [DirtyCpu] Rust: model.create_session(SessionParams)
      → Wraps LlamaSession in Mutex, then ResourceArc<ExLLamaSessionRef>
    → Returns ExLLamaSession{model_name, seed, resource}
  → Returns %ExLLama.Session{seed, model_name, resource: ref}
```

## Synchronous Completion

```
User: ExLLama.completion(session, max_tokens, stop)
  → ExLLama.Nif.__session_nif_completion__(resource, max_tokens, stop)
    → [DirtyCpu] Rust:
      1. Lock session mutex
      2. Deep copy session (avoids holding lock during inference)
      3. Start completing with StandardSampler
      4. Collect string tokens, apply optional regex stop pattern
      5. Return ExLLamaCompletion{content, token_length}
  → Returns {:ok, %{content: string, token_length: int}}
```

## Streaming Completion

```
User: ExLLama.Session.start_completing_with(session, opts)
  → Spawns Elixir process that calls:
    ExLLama.Nif.__session_nif_start_completing_with__(pid, resource, max_tokens)
      → [DirtyCpu] Rust:
        1. Lock session, deep copy, release lock
        2. Start completing with StandardSampler
        3. For each token string: env.send(&pid, completion)
        4. Send :fin atom when done
  → Caller receives stream of string messages, then :fin
```

## Chat Completion

```
User: ExLLama.chat_completion(model, thread, options)
  1. Build SessionOptions (with seed)
  2. Create session
  3. ChatTemplate.to_context(thread, model, options) → prompt string
  4. Session.set_context(session, prompt)
  5. For each choice (1..N):
     Session.completion(session, max_tokens, nil) → {token_count, text}
  6. ChatTemplate.extract_response(choices, model, options)
     → Returns GenAI.ChatCompletion struct
```

## Resource Lifecycle

- **Model**: `ResourceArc<ExLLamaModelRef>` — ref-counted, freed when all Elixir references are GC'd
- **Session**: `ResourceArc<ExLLamaSessionRef>` wrapping `Mutex<LlamaSession>` — same GC-based lifecycle
- Sessions hold an implicit reference to their parent model (via llama_cpp internals)
