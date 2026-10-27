# c_src/ — Native C NIF Layer

```
c_src/
├── ex_llama_nif.cpp                # Single translation unit: resources + all NIF functions
├── jev_head.h                      # JEVH0001 decision-head format: parse + normalize + forward
├── test_jev_head.cpp               # Portable decision-head unit test (no Erlang/CUDA/model)
├── test_llama_nif.cpp              # Standalone NIF smoke binary (no Erlang runtime needed)
└── llama.cpp/                      # llama.cpp submodule, pinned v0.6.0 (vendored — do not annotate)
```

## Resources (`ex_llama_nif.cpp`)

| Resource type | Wraps | Lifetime |
|---------------|-------|----------|
| `MODEL_RES_TYPE` → `ModelRes` | `llama_model*` (+ `classifier_mu`) | freed in destructor via `llama_model_free` |
| `CTX_RES_TYPE` → `CtxRes` | `llama_context*` (+ non-owning `llama_model*`, `std::mutex`) | freed via `llama_free` |
| `HEAD_RES_TYPE` → `HeadRes` | `ex_llama::JevHead` (FP32 weights/mu/sd/bias) | C++ destructor |

## Exposed NIF functions (from the `nif_funcs[]` table)

Model / vocabulary (dirty-CPU unless noted):
`load_model/1,2`, `create_context/1,2`, `context_deep_copy/1`, `tokenize/2,3`,
`detokenize/2`, `token_to_piece/2`, `vocab_bos|eos|eot|nl/1`,
`vocab_fim_pre|mid|suf/1`, `model_info/1`.

Session / inference:
`completion/2,3` (dirty-CPU), `streaming_completion/3,4`,
`embeddings/2` (dirty-CPU), `chat_apply_template/2,3`.

Classifier:
`classifier_load_head/1` (dirty-CPU), `classifier_head_info/1`,
`classifier_head_probabilities/3` (dirty-CPU), `classifier_last_hidden/3` (dirty-CPU).

Registered with `ERL_NIF_INIT(Elixir.ExLLama.Nif, …)`; `on_load` is the only
lifecycle callback.

## Build

`Makefile` (invoked by `elixir_make` during `mix compile`) builds the pinned
`c_src/llama.cpp` with cmake into static archives, then links
`c_src/ex_llama_nif.cpp` into `priv/ex_llama_nif.so`. Metal frameworks are
linked on Darwin; `-lgomp` on Linux. `LLAMA_BUILD_COMMON/TOOLS/APP/TESTS/
EXAMPLES/SERVER` are all forced OFF (v0.6.0 defaults them ON).
