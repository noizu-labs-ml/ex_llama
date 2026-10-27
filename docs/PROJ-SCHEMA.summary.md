# Project Schema Summary

**No relational persistence.** ExLLama owns no database, Ecto schemas, or
Liquibase changelogs. Its data is in-memory option structs, two binary file
formats, process-dictionary session state, and GenAI interface structs.

## In-Memory Structs (`lib/ex_llama/`)
- **ModelOptions** — n_gpu_layers, split_mode, main_gpu, load_mode
  (auto/none/mmap/mlock/mmap+mlock/direct_io), lazy_mode, booleans;
  deprecated use_mmap/use_mlock fold into load_mode.
- **SessionOptions / ContextParams** — seed, n_ctx, n_batch, n_ubatch, n_seq_max,
  threads, rope/yarn params, type_k/type_v (ggml ints; 1=F16,8=Q8_0,2=Q4_0,30=BF16),
  flash_attn/pooling/attention types, op_offload, kv_unified, swa_full.
- **EmbeddingOptions** — n_threads, n_threads_batch.

## Binary Formats
- **GGUF** — external llama.cpp model format (architectures per v0.6.0 pin).
- **JEVH0001** — little-endian FP32 decision head: magic, uint32 hidden, uint32
  classes(=256), mu[hidden], sd[hidden], weights[hidden×classes], bias[classes];
  size = 16 + (2h + h·c + c)·4, ≤128 MiB; strictly validated, never a pickle.

## Session State (process dictionary, keyed by ctx ref)
{:ex_llama_ctx,ref} text · {:ex_llama_seed,ref} · {:ex_llama_model,ref} ·
{:ex_llama_params,ref}.

## Interfaces
- Completion: {:ok, %{content, token_length}}; streaming → token msgs + :fin.
- ChatTemplate: support_list/0, to_context/3, extract_response/3 → GenAI.ChatCompletion (default Zephyr).
- Classifier: choice/boolean/score, text-only, ≤256 options, native_multi_question false,
  returns GenAI.Classifier.Result with normalized distributions.
