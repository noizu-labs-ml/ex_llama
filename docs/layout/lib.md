# lib/ — Elixir Source

```
lib/
├── ex_llama/
│   ├── chat_template/              # Chat format implementations (behaviour-based)
│   │   ├── alpaca.ex               #   Alpaca format
│   │   ├── amber_chat.ex           #   AmberChat format
│   │   ├── chat_ml.ex              #   ChatML format
│   │   ├── chat_qa.ex              #   ChatQA format
│   │   ├── falcon_instruct.ex      #   Falcon Instruct format
│   │   ├── gemma_instruct.ex       #   Gemma Instruct format
│   │   ├── granite_instruct.ex     #   Granite Instruct format
│   │   ├── llama_2_chat.ex         #   Llama 2 Chat format
│   │   ├── llama_3_instruct.ex     #   Llama 3 Instruct format
│   │   ├── mistral_instruct.ex     #   Mistral Instruct format
│   │   ├── open_chat.ex            #   OpenChat format
│   │   ├── phi_3.ex                #   Phi-3 format
│   │   ├── phi_3_small.ex          #   Phi-3 Small format
│   │   ├── qwen_instruct.ex        #   Qwen Instruct format
│   │   ├── saiga.ex                #   Saiga format
│   │   ├── solar_instruct.ex       #   Solar Instruct format
│   │   ├── vicuna.ex               #   Vicuna format
│   │   └── zephyr.ex               #   Zephyr format (default)
│   ├── chat_template.ex            # ChatTemplate behaviour + pick_handler/2
│   ├── classifier.ex               # GenAI.Classifier.ProviderBehaviour impl (Jev-Omni)
│   ├── classifier/
│   │   └── decision.ex             # Pure distribution → GenAI Answer mapping
│   ├── context_params.ex           # Context parameter struct (mirrors SessionOptions)
│   ├── embedding_options.ex        # Embedding options struct (thread counts)
│   ├── model_options.ex            # Model loading options struct (+ load_mode resolution)
│   ├── model.ex                    # Model handle over the native model resource
│   ├── nif.ex                      # C NIF bindings + bridge shims (old API → C NIFs)
│   ├── session_options.ex          # Session options struct
│   └── session.ex                  # Inference session over the native context resource
└── ex_llama.ex                     # Main API — load_model, create_session, chat_completion, classify
```

## Notes

- `ExLLama.Nif` holds both the raw C NIF stubs (`:nif_not_loaded`) and the
  `__model_nif_*__` / `__session_nif_*__` bridge functions that translate the
  historical Rust-era API onto the current C NIF surface and wrap raw
  references in `%ExLLama.Model{}` / `%ExLLama.Session{}`.
- Context text, seed, model, and params are stashed in the calling process's
  dictionary keyed by the native context ref (see `ExLLama.Session` and
  `ExLLama.Nif`), so a session created in one process can be driven from another.
- `ExLLama.ContextParams` duplicates the `SessionOptions` field set for callers
  that prefer the context-terminology struct.
