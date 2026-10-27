# Data Flow

## Model Loading

```
User: ExLLama.load_model(path, %ModelOptions{})
  → ExLLama.Model.load_from_file/2
    → ExLLama.Nif.__model_nif_load_from_file__(path, opts)
      → [DirtyCpu] C NIF load_model/2 → ModelRes{ llama_model* }
      → vocab_bos / vocab_eos / token_to_piece to derive bos/eos pieces
    → %ExLLama.Model{resource: ref, name: path, eos: charlist, bos: charlist}
```

`ModelOptions` is translated in the bridge: only non-nil fields are forwarded
(`load_mode`, `n_gpu_layers`, `split_mode`, `main_gpu`, `lazy_mode`, and the
boolean flags); deprecated `use_mmap`/`use_mlock` are resolved into `load_mode`.

## Session Creation

```
User: ExLLama.create_session(model, options)
  → ExLLama.Nif.__model_nif_create_session__(model, options)
    → C NIF create_context/2 → CtxRes{ llama_context*, model, mutex }
    → seed, model, and params stashed in the caller's process dictionary
      under {:ex_llama_seed|model|params, ctx_ref}
    → %ExLLama.Session{resource: ref, model_name, seed}
```

`seed` is not a `llama_context_params` member upstream; it is stashed and
re-applied at sampling time on each completion/embedding call.

## Synchronous Completion

```
User: ExLLama.completion(session, max_tokens, stop)
  → ExLLama.Nif.__session_nif_completion__(ctx_ref, max_tokens, stop)
    → reads {:ex_llama_ctx, ctx_ref} + {:ex_llama_seed, ctx_ref} from process dict
    → [DirtyCpu] C NIF completion/3 (prompt, %{max_tokens, seed, stop})
    → {:ok, %{content: string, token_length: int}}
```

## Streaming Completion

```
User: ExLLama.Session.start_completing_with(session, %{max_tokens: n, pid: pid})
  → captures context/seed into the process dict, ensure_seed/1
  → C NIF streaming_completion/4 (ctx_ref, prompt, pid, opts)
  → caller receives token messages, then :fin
```

## Chat Completion

```
User: ExLLama.chat_completion(model, thread, options)
  1. Build SessionOptions (seed from options[:seed] / options[:session_options])
  2. create_session
  3. ChatTemplate.to_context(thread, model, options) → prompt string
  4. Session.set_context(session, prompt)
  5. For each choice (1..n): Session.completion(...) → {token_count, text}
  6. ChatTemplate.extract_response(choices, model, options) → GenAI.ChatCompletion
```

## Classifier

```
User: ExLLama.classify(classifier, request)
  → GenAI.classify → ExLLama.Classifier.classify/2
    1. Request.normalize; reject non-text modality
    2. For each question: build Jev prompt (prompt/3), one backbone forward pass
       via classifier_last_hidden/3 → last-token hidden features
    3. classifier_head_probabilities/3 → normalized distribution (mu/sd + linear head)
    4. Classifier.Decision.answer/2 → GenAI Answer (value/index/distribution/probability)
  → %GenAI.Classifier.Result{}
```

## Resource Lifecycle

- **Model / Context / Head**: NIF resources; destructors free the llama.cpp model,
  context, or clear the C++ head. Freed when the BEAM reference is GC'd.
- A model's `classifier_mu` serialises concurrent classifier calls on that model.
- A context resource is non-owning of its model: keep the model alive while the
  session is in use.
