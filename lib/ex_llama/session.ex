defmodule ExLLama.Session do
  defstruct [
    seed: nil,
    model_name: nil,
    resource: nil
  ]

  # Sessions can be created in one process and used from another (e.g. handed
  # to a GenServer or Task). The struct is the durable seed carrier; the
  # NIF-side bridges read a process-dictionary stash keyed by the context ref.
  # (Re)populate the calling process's stash from the struct so completion,
  # streaming, params and deep_copy see the seed regardless of which process
  # created the session. 0xFFFFFFFF is llama.cpp's LLAMA_DEFAULT_SEED, so
  # stashing it for never-seeded sessions is behaviourally a no-op.
  defp ensure_seed(%__MODULE__{resource: ref, seed: seed}) when seed != nil do
    unless Process.get({:ex_llama_seed, ref}) do
      Process.put({:ex_llama_seed, ref}, seed)
    end
    :ok
  end

  # <REMOVED UUID HERE> default_options :: auto-generated pointer for public function default_options
  def default_options(), do: ExLLama.Nif.__session_nif_default_session_options__()
  # <REMOVED UUID HERE> advance_context_with_tokens :: auto-generated pointer for public function advance_context_with_tokens
  def advance_context_with_tokens(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context_with_tokens__(session.resource, context)
  # <REMOVED UUID HERE> advance_context :: auto-generated pointer for public function advance_context
  def advance_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context__(session.resource, context)
  # <REMOVED UUID HERE> start_completing_with :: auto-generated pointer for public function start_completing_with
  def start_completing_with(%__MODULE__{resource: _} = session, options) do
    ensure_seed(session)
    max_tokens = options[:max_tokens] || 512
    pid = options[:pid] || self()
    # Capture context from process dict before spawning
    prompt = Process.get({:ex_llama_ctx, session.resource}, "")
    seed = Process.get({:ex_llama_seed, session.resource})
    opts = %{max_tokens: max_tokens}
    opts = if seed, do: Map.put(opts, :seed, seed), else: opts
    ExLLama.Nif.streaming_completion(session.resource, prompt, pid, opts)
    :ok
  end
  # <REMOVED UUID HERE> completion :: auto-generated pointer for public function completion
  def completion(%__MODULE__{resource: _} = session, max_tokens, stop) do
    ensure_seed(session)
    ExLLama.Nif.__session_nif_completion__(session.resource, max_tokens, stop)
  end
  # <REMOVED UUID HERE> model :: auto-generated pointer for public function model
  def model(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_model__(session.resource)
  # <REMOVED UUID HERE> params :: auto-generated pointer for public function params
  def params(%__MODULE__{resource: _} = session) do
    ensure_seed(session)
    ExLLama.Nif.__session_nif_params__(session.resource)
  end
  # <REMOVED UUID HERE> context_size :: auto-generated pointer for public function context_size
  def context_size(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context_size__(session.resource)
  # <REMOVED UUID HERE> context :: auto-generated pointer for public function context
  def context(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context__(session.resource)
  # <REMOVED UUID HERE> truncate_context :: auto-generated pointer for public function truncate_context
  def truncate_context(%__MODULE__{resource: _} = session, n_tokens), do: ExLLama.Nif.__session_nif_truncate_context__(session.resource, n_tokens)
  # <REMOVED UUID HERE> set_context_to_tokens :: auto-generated pointer for public function set_context_to_tokens
  def set_context_to_tokens(%__MODULE__{resource: _} = session, tokens), do: ExLLama.Nif.__session_nif_set_context_to_tokens__(session.resource, tokens)
  # <REMOVED UUID HERE> set_context :: auto-generated pointer for public function set_context
  def set_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_set_context__(session.resource, context)
  # <REMOVED UUID HERE> deep_copy :: auto-generated pointer for public function deep_copy
  def deep_copy(%__MODULE__{resource: _} = session) do
    ensure_seed(session)
    with {:ok, copy} <- ExLLama.Nif.__session_deep_copy__(session.resource) do
      {:ok, put_in(copy, [Access.key(:model_name)], session.model_name)}
    end
  end

end
