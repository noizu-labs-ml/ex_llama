# Project Layout Summary

```
ex_llama/
├── lib/                            # Elixir source
│   ├── ex_llama/                   # Submodules
│   │   ├── chat_template/          # 18 chat format implementations
│   │   ├── chat_template.ex        # Behaviour definition
│   │   ├── model.ex                # Model management
│   │   ├── session.ex              # Session handling
│   │   └── nif.ex                  # NIF bindings
│   └── ex_llama.ex                 # Main API
├── native/erlang_llama_cpp_nif/    # Rust NIF (Rustler + llama_cpp)
│   └── src/                        # nifs/, refs/, structs/
├── chat_templates/                 # Jinja2 templates + generation configs
├── test/                           # Tests
├── priv/                           # Models + compiled NIFs
├── mix.exs                         # Project config
└── CLAUDE.md                       # Claude Code instructions
```
