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
│   ├── chat_template.ex            # ChatTemplate behaviour definition
│   ├── context_params.ex           # Context parameter struct
│   ├── embedding_options.ex        # Embedding options struct
│   ├── model_options.ex            # Model loading options struct
│   ├── model.ex                    # Model loading and management
│   ├── nif.ex                      # NIF bindings (Rustler, stubs for all NIF functions)
│   ├── session_options.ex          # Session options struct
│   └── session.ex                  # Inference session handling
└── ex_llama.ex                     # Main API — convenience functions, chat completion
```
