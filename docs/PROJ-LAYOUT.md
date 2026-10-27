# Project Layout

```
ex_llama/
├── lib/                            # Elixir source → [layout/lib.md](layout/lib.md)
│   ├── ex_llama.ex                 #   Main public API (ExLLama)
│   └── ex_llama/                   #   Submodules (Model, Session, Nif, ChatTemplate, Classifier, option structs)
├── c_src/                          # Native C/C++ NIF layer → [layout/native.md](layout/native.md)
│   ├── ex_llama_nif.cpp            #   C NIF: resources + all NIF functions
│   ├── jev_head.h                  #   JEVH0001 FP32 decision-head parser/forward pass
│   ├── test_llama_nif.cpp          #   Standalone (no-Erlang) NIF smoke binary
│   ├── test_jev_head.cpp           #   Standalone decision-head unit test
│   └── llama.cpp/                  #   llama.cpp submodule (pinned v0.6.0), vendored, not annotated
├── scripts/                        # Classifier reference/export tooling
│   ├── export_jev_head.py          #   Export reference decision head to JEVH0001
│   ├── compare_jev_parity.py       #   Reference vs native parity check
│   ├── run_jev_reference.py        #   Run upstream reference classifier
│   ├── test_export_jev_head.py     #   Exporter unit test
│   └── validate_jev_classifier.exs #   Live classifier validation script
├── test/                           # Test suite (ExUnit)
│   ├── support/quiet_logger.ex     #   Logger-silencing helper
│   ├── ex_llama_test.exs           #   Model/session/completion tests
│   ├── chat_template_test.exs      #   Chat template formatting tests
│   ├── ex_llama/kitchen_sink_test.exs          # Session/completion default sweep
│   ├── ex_llama/classifier_contract_test.exs   # Core classifier contract (pure)
│   ├── ex_llama/classifier_native_test.exs     # Native classifier integration
│   └── test_helper.exs             #   Test bootstrap
├── docs/                           # Documentation
│   ├── PROJ-LAYOUT.md / .summary.md
│   ├── PROJ-ARCH.md / .summary.md
│   ├── PROJ-SCHEMA.md / .summary.md
│   ├── THREAT-MODEL.md / .summary.md
│   ├── jev-classifier.md           #   Native classifier setup + validation
│   ├── llama-cpp-feature-audit.md  #   0.4.0 llama.cpp feature delta catalog
│   ├── arch/                       #   Architecture detail docs
│   └── layout/                     #   Layout detail docs
├── priv/                           # Runtime artifacts
│   ├── models/local_llama/         #   Test models (TinyLlama GGUF, gitignored)
│   └── ex_llama_nif.so             #   Compiled NIF (built by elixir_make, gitignored)
├── .github/workflows/ci.yml        # CI: compile (warnings-as-errors) + test on ubuntu-22.04
├── .tool-versions                  # asdf versions (Erlang 28.4, Elixir 1.19.5)
├── .gitmodules                     # c_src/llama.cpp submodule
├── Makefile                        # llama.cpp cmake + NIF link driver (elixir_make)
├── mix.exs                         # Project config, deps, build settings
├── mix.lock                        # Locked dependency versions
├── AGENTS.md / AGENT.md            # Agent guidance (kept aligned)
├── CLAUDE.md                       # Claude Code project instructions
├── CHANGELOG.md                    # Version history
├── LICENSE                         # MIT license
└── README.md                       # Project overview
```

Note: the `native/` Rust crate and the `chat_templates/` Jinja reference directory
described by earlier revisions of this doc no longer exist — the native layer was
rewritten as a C NIF (CHANGELOG 0.2.0) and the chat templates are now first-class
Elixir modules under `lib/ex_llama/chat_template/`.

## Key Files Requiring Setup

| File | Action |
|------|--------|
| `.tool-versions` | Install runtimes via `asdf install` (Elixir 1.19.5 / OTP 28.4) |
| `c_src/llama.cpp/` | Populate submodule: `git submodule update --init` (Makefile clones pinned `v0.6.0` if absent) |
| `priv/models/local_llama/tiny_llama/` | Download the TinyLlama GGUF test model before running model-backed tests |
| cmake + C++ toolchain | Required at compile time to build the native NIF |
