# native/ — Rust NIF Layer

```
native/
└── erlang_llama_cpp_nif/
    ├── .cargo/
    │   └── config.toml             # Cargo build config
    ├── src/
    │   ├── nifs/
    │   │   ├── ex_llama_model.rs   # Model NIF functions (load, tokenize, decode, etc.)
    │   │   └── ex_llama_session.rs # Session NIF functions (advance, complete, context ops)
    │   ├── refs/
    │   │   ├── model_ref.rs        # Model resource reference (Rustler ResourceArc)
    │   │   └── session_ref.rs      # Session resource reference (Rustler ResourceArc)
    │   ├── structs/
    │   │   ├── completion.rs       # Completion result struct
    │   │   ├── embedding_options.rs# Embedding options struct
    │   │   ├── model_options.rs    # Model options struct
    │   │   ├── model.rs            # Model struct
    │   │   ├── session_options.rs  # Session options struct
    │   │   └── session.rs          # Session struct
    │   ├── lib.rs                  # Crate root, Rustler NIF init macro
    │   ├── nifs.rs                 # NIF module declarations
    │   ├── refs.rs                 # Resource reference module declarations
    │   └── structs.rs              # Struct module declarations
    ├── Cargo.toml                  # Crate deps (rustler, llama_cpp crate)
    └── Cargo.lock                  # Locked Rust deps
```

## NIF Functions Exposed

### Model NIFs
- `__model_nif_load_from_file__/2` — Load GGUF model from path
- `__model_nif_detokenize__/2` — Convert tokens back to text
- `__model_nif_token_to_byte_piece__/2` — Single token to bytes
- `__model_nif_token_to_piece__/2` — Single token to string
- `__model_nif_decode_tokens__/2` — Decode token list
- `__model_nif_create_session__/2` — Create inference session
- `__model_nif_embeddings__/3` — Generate embeddings
- `__model_nif_bos__/1`, `eos/1`, `nl/1` — Special token IDs
- `__model_nif_infill_prefix__/1`, `middle/1`, `suffix/1`, `eot/1` — Infill tokens
- `__model_nif_vocabulary_size__/1`, `embed_len/1`, `train_len/1` — Model metadata

### Session NIFs
- `__session_nif_advance_context__/2` — Advance context with text
- `__session_nif_advance_context_with_tokens__/2` — Advance with token IDs
- `__session_nif_completion__/3` — Synchronous completion
- `__session_nif_start_completing_with__/3` — Async/streaming completion
- `__session_nif_model__/1` — Get session's model reference
- `__session_nif_params__/1` — Get session parameters
- `__session_nif_context_size__/1` — Current context length
- `__session_nif_context__/1` — Get context tokens
- `__session_nif_truncate_context__/2` — Truncate context to length
- `__session_nif_set_context_to_tokens__/2` — Replace context with tokens
- `__session_nif_set_context__/2` — Replace context with text
- `__session_deep_copy__/1` — Deep copy session state
