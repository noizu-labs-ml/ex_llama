# Project Layout Summary

```
ex_llama/
├── lib/
│   ├── ex_llama.ex                 # Main public API
│   └── ex_llama/
│       ├── chat_template/          # 18 chat-format implementations
│       ├── chat_template.ex        # Behaviour + pick_handler
│       ├── classifier.ex           # GenAI classifier provider
│       ├── classifier/decision.ex  # Pure distribution → answer mapping
│       ├── model.ex / session.ex   # Model + session handles over NIF refs
│       ├── nif.ex                  # C NIF bindings + bridge shims
│       └── *_options.ex / context_params.ex  # Option structs
├── c_src/                          # Native C/C++ NIF
│   ├── ex_llama_nif.cpp            # C NIF (resources, ~30 NIF functions)
│   ├── jev_head.h                  # JEVH0001 decision-head parser
│   ├── test_*.cpp                  # Standalone native tests
│   └── llama.cpp/                  # llama.cpp submodule (v0.6.0)
├── scripts/                        # Classifier export/parity/validation tooling
├── test/                           # ExUnit suite
├── docs/                           # PROJ-*, THREAT-MODEL, arch/, layout/, jev-classifier.md
├── priv/                           # Test models + built NIF (.so gitignored)
├── .github/workflows/ci.yml        # CI: compile + test
├── Makefile                        # llama.cpp cmake + NIF link
├── mix.exs                         # Project config
└── README.md                       # Start here
```
