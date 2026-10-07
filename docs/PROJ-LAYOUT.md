# Project Layout

```
ex_llama/
├── lib/                            # Elixir source → [layout/lib.md](layout/lib.md)
│   ├── ex_llama/                   #   Submodules (Model, Session, Nif, ChatTemplate)
│   └── ex_llama.ex                 #   Main API module
├── native/                         # Rust NIF layer → [layout/native.md](layout/native.md)
│   └── erlang_llama_cpp_nif/       #   Rustler crate wrapping llama_cpp
├── chat_templates/                 # Jinja2 chat templates + generation configs
│   ├── chat_templates/             #   .jinja template files (18 formats)
│   └── generation_configs/         #   .json generation config files (19 configs)
├── test/                           # Test suite
│   ├── support/                    #   Test helpers (quiet_logger.ex)
│   ├── ex_llama_test.exs           #   Core model/session/completion tests
│   ├── chat_template_test.exs      #   Chat template formatting tests
│   └── test_helper.exs             #   Test bootstrap
├── priv/                           # Runtime artifacts
│   ├── models/local_llama/         #   Test models (TinyLlama GGUF)
│   └── native/                     #   Compiled NIF .so files (gitignored)
├── doc/                            # Generated ExDoc output (gitignored)
├── .tool-versions                  # asdf versions (Erlang 28.4, Elixir 1.19.5)
├── CLAUDE.md                       # Claude Code project instructions
├── CHANGELOG.md                    # Version history
├── mix.exs                         # Project config, deps, build settings
├── mix.lock                        # Locked dependency versions
├── LICENSE                         # MIT license
└── README.md                       # Project overview
```

## Key Files Requiring Setup

| File | Action |
|------|--------|
| `.tool-versions` | Install runtimes via `asdf install` |
| `priv/models/local_llama/tiny_llama/` | Run `bash init.sh` to download test model |
