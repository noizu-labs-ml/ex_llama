defmodule ExLLama.Nif do
  @moduledoc """
  C NIF bindings for llama.cpp — bridges old API names to new C NIF functions
  and wraps raw references in the expected structs.
  """

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:ex_llama), ~c"ex_llama_nif")
    :erlang.load_nif(path, 0)
  end

  # ---- Raw NIF functions (loaded from C) ----
  def load_model(_path), do: :erlang.nif_error(:nif_not_loaded)
  def load_model(_path, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def create_context(_model, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def create_context(_model), do: :erlang.nif_error(:nif_not_loaded)
  def context_deep_copy(_ctx), do: :erlang.nif_error(:nif_not_loaded)
  def tokenize(_model, _text), do: :erlang.nif_error(:nif_not_loaded)
  def tokenize(_model, _text, _add_special), do: :erlang.nif_error(:nif_not_loaded)
  def detokenize(_model, _tokens), do: :erlang.nif_error(:nif_not_loaded)
  def token_to_piece(_model, _token), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_bos(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_eos(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_eot(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_nl(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_fim_pre(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_fim_mid(_model), do: :erlang.nif_error(:nif_not_loaded)
  def vocab_fim_suf(_model), do: :erlang.nif_error(:nif_not_loaded)
  def model_info(_model), do: :erlang.nif_error(:nif_not_loaded)
  def completion(_ctx, _prompt), do: :erlang.nif_error(:nif_not_loaded)
  def completion(_ctx, _prompt, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def streaming_completion(_ctx, _prompt, _pid), do: :erlang.nif_error(:nif_not_loaded)
  def streaming_completion(_ctx, _prompt, _pid, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def embeddings(_ctx, _text), do: :erlang.nif_error(:nif_not_loaded)
  def chat_apply_template(_model, _messages), do: :erlang.nif_error(:nif_not_loaded)
  def chat_apply_template(_model, _messages, _add_gen), do: :erlang.nif_error(:nif_not_loaded)

  # ---- Bridging functions that return expected structs ----

  def __model_nif_load_from_file__(path, %ExLLama.ModelOptions{} = opts) do
    nif_opts = %{
      n_gpu_layers: opts.n_gpu_layers || 0,
      vocab_only: opts.vocab_only || false,
      use_mmap: opts.use_mmap || false,
      use_mlock: opts.use_mlock || false
    }
    case load_model(path, nif_opts) do
      {:ok, ref} ->
        {:ok, bos} = vocab_bos(ref)
        {:ok, eos} = vocab_eos(ref)
        {:ok, bos_piece} = token_to_piece(ref, bos)
        {:ok, eos_piece} = token_to_piece(ref, eos)
        {:ok, %ExLLama.Model{
          resource: ref,
          name: path,
          bos: String.to_charlist(bos_piece),
          eos: String.to_charlist(eos_piece)
        }}
      error -> error
    end
  end

  def __model_nif_create_session__(model, options) do
    o = if is_struct(options), do: Map.from_struct(options), else: Map.new(options || [])
    ctx_opts = %{}
    ctx_opts = if o[:n_ctx], do: Map.put(ctx_opts, :n_ctx, o[:n_ctx]), else: ctx_opts
    ctx_opts = if o[:n_batch], do: Map.put(ctx_opts, :n_batch, o[:n_batch]), else: ctx_opts
    ctx_opts = if o[:n_threads], do: Map.put(ctx_opts, :n_threads, o[:n_threads]), else: ctx_opts
    ctx_opts = if o[:n_threads_batch], do: Map.put(ctx_opts, :n_threads_batch, o[:n_threads_batch]), else: ctx_opts
    ctx_opts = if o[:rope_freq_base], do: Map.put(ctx_opts, :rope_freq_base, o[:rope_freq_base] / 1), else: ctx_opts
    ctx_opts = if o[:rope_freq_scale], do: Map.put(ctx_opts, :rope_freq_scale, o[:rope_freq_scale] / 1), else: ctx_opts
    ctx_opts = if o[:embedding], do: Map.put(ctx_opts, :embeddings, o[:embedding]), else: ctx_opts
    ctx_opts = if o[:offload_kqv] != nil, do: Map.put(ctx_opts, :offload_kqv, o[:offload_kqv]), else: ctx_opts

    case create_context(model.resource, ctx_opts) do
      {:ok, ctx_ref} ->
        {:ok, %ExLLama.Session{
          resource: ctx_ref,
          model_name: model.name,
          seed: o[:seed] || 0xFFFFFFFF
        }}
      error -> error
    end
  end

  def __model_nif_detokenize__(model, token), do: token_to_piece(model.resource, token)
  def __model_nif_token_to_byte_piece__(model, token), do: token_to_piece(model.resource, token)
  def __model_nif_token_to_piece__(model, token), do: token_to_piece(model.resource, token)
  def __model_nif_decode_tokens__(model, tokens), do: detokenize(model.resource, tokens)

  def __model_nif_embeddings__(model, inputs, _options) do
    case create_context(model.resource, %{embeddings: true}) do
      {:ok, ctx_ref} -> embeddings(ctx_ref, inputs)
      error -> error
    end
  end

  def __model_nif_bos__(model), do: vocab_bos(model.resource)
  def __model_nif_eos__(model), do: vocab_eos(model.resource)
  def __model_nif_nl__(model), do: vocab_nl(model.resource)
  def __model_nif_infill_prefix__(model), do: vocab_fim_pre(model.resource)
  def __model_nif_infill_middle__(model), do: vocab_fim_mid(model.resource)
  def __model_nif_infill_suffix__(model), do: vocab_fim_suf(model.resource)
  def __model_nif_eot__(model), do: vocab_eot(model.resource)

  def __model_nif_vocabulary_size__(model) do
    case model_info(model.resource) do
      {:ok, info} -> {:ok, info.n_vocab}
      error -> error
    end
  end

  def __model_nif_embed_len__(model) do
    case model_info(model.resource) do
      {:ok, info} -> {:ok, info.n_embd}
      error -> error
    end
  end

  def __model_nif_train_len__(model) do
    case model_info(model.resource) do
      {:ok, info} -> {:ok, info.n_ctx_train}
      error -> error
    end
  end

  # ---- Session NIF bridges ----
  # Context text is stored in process dictionary keyed by ctx_ref.
  # Model ref and seed are also stored for tokenization and completion.

  def __session_nif_default_session_options__() do
    {:ok, %ExLLama.SessionOptions{
      seed: 0xFFFFFFFF,
      n_ctx: 2048,
      n_batch: 512,
      n_threads: 4,
      n_threads_batch: 4,
      rope_scaling_type: 0,
      rope_freq_base: 0.0,
      rope_freq_scale: 0.0,
      yarn_ext_factor: 0.0,
      yarn_attn_factor: 0.0,
      yarn_beta_fast: 0.0,
      yarn_beta_slow: 0.0,
      yarn_orig_ctx: 0,
      type_k: 0,
      type_v: 0,
      embedding: false,
      offload_kqv: true,
      pooling: false
    }}
  end

  def __session_nif_advance_context__(ctx_ref, context) when is_binary(context) do
    existing = Process.get({:ex_llama_ctx, ctx_ref}, "")
    Process.put({:ex_llama_ctx, ctx_ref}, existing <> context)
    {:ok, String.length(context)}
  end

  def __session_nif_advance_context_with_tokens__(ctx_ref, tokens) when is_list(tokens) do
    model = Process.get({:ex_llama_model, ctx_ref})
    case detokenize(model.resource, tokens) do
      {:ok, text} ->
        existing = Process.get({:ex_llama_ctx, ctx_ref}, "")
        Process.put({:ex_llama_ctx, ctx_ref}, existing <> text)
        {:ok, length(tokens)}
      error -> error
    end
  end

  def __session_nif_set_context__(ctx_ref, context) when is_binary(context) do
    Process.put({:ex_llama_ctx, ctx_ref}, context)
    {:ok, :ok}
  end

  def __session_nif_set_context_to_tokens__(ctx_ref, tokens) when is_list(tokens) do
    model = Process.get({:ex_llama_model, ctx_ref})
    case detokenize(model.resource, tokens) do
      {:ok, text} ->
        Process.put({:ex_llama_ctx, ctx_ref}, text)
        {:ok, :ok}
      error -> error
    end
  end

  def __session_nif_context__(ctx_ref) do
    text = Process.get({:ex_llama_ctx, ctx_ref}, "")
    model = Process.get({:ex_llama_model, ctx_ref})
    if model do
      tokenize(model.resource, text, false)
    else
      {:ok, text}
    end
  end

  def __session_nif_context_size__(ctx_ref) do
    case __session_nif_context__(ctx_ref) do
      {:ok, tokens} when is_list(tokens) -> {:ok, length(tokens)}
      {:ok, text} when is_binary(text) -> {:ok, String.length(text)}
      error -> error
    end
  end

  def __session_nif_truncate_context__(ctx_ref, n_tokens) do
    model = Process.get({:ex_llama_model, ctx_ref})
    text = Process.get({:ex_llama_ctx, ctx_ref}, "")
    if model do
      # Token-level truncation: tokenize, take first n, detokenize
      case tokenize(model.resource, text, false) do
        {:ok, tokens} ->
          truncated = Enum.take(tokens, n_tokens)
          case detokenize(model.resource, truncated) do
            {:ok, new_text} ->
              Process.put({:ex_llama_ctx, ctx_ref}, new_text)
              {:ok, :ok}
            error -> error
          end
        error -> error
      end
    else
      Process.put({:ex_llama_ctx, ctx_ref}, String.slice(text, 0, n_tokens))
      {:ok, :ok}
    end
  end

  def __session_nif_completion__(ctx_ref, max_tokens, stop) do
    prompt = Process.get({:ex_llama_ctx, ctx_ref}, "")
    seed = Process.get({:ex_llama_seed, ctx_ref})
    opts = %{max_tokens: max_tokens}
    opts = if seed, do: Map.put(opts, :seed, seed), else: opts
    opts = if stop, do: Map.put(opts, :stop, stop), else: opts
    completion(ctx_ref, prompt, opts)
  end

  def __session_nif_start_completing_with__(pid, ctx_ref, max_tokens) do
    prompt = Process.get({:ex_llama_ctx, ctx_ref}, "")
    seed = Process.get({:ex_llama_seed, ctx_ref})
    opts = %{max_tokens: max_tokens}
    opts = if seed, do: Map.put(opts, :seed, seed), else: opts
    streaming_completion(ctx_ref, prompt, pid, opts)
  end

  def __session_nif_model__(ctx_ref) do
    case Process.get({:ex_llama_model, ctx_ref}) do
      nil -> {:error, "no model associated with this session"}
      model -> {:ok, model}
    end
  end

  def __session_nif_params__(ctx_ref) do
    case Process.get({:ex_llama_params, ctx_ref}) do
      nil ->
        {:ok, defaults} = __session_nif_default_session_options__()
        seed = Process.get({:ex_llama_seed, ctx_ref})
        {:ok, if(seed, do: %{defaults | seed: seed}, else: defaults)}
      params -> {:ok, params}
    end
  end

  def __session_deep_copy__(ctx_ref) do
    model = Process.get({:ex_llama_model, ctx_ref})
    seed = Process.get({:ex_llama_seed, ctx_ref})
    context = Process.get({:ex_llama_ctx, ctx_ref}, "")
    params = Process.get({:ex_llama_params, ctx_ref})

    # Deep copy with KV cache state preservation
    case context_deep_copy(ctx_ref) do
      {:ok, new_ref} ->
        Process.put({:ex_llama_model, new_ref}, model)
        if seed, do: Process.put({:ex_llama_seed, new_ref}, seed)
        Process.put({:ex_llama_ctx, new_ref}, context)
        if params, do: Process.put({:ex_llama_params, new_ref}, params)
        {:ok, %ExLLama.Session{
          resource: new_ref,
          model_name: model.name,
          seed: seed || 0xFFFFFFFF
        }}
      error -> error
    end
  end
end
