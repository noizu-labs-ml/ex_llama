defmodule ExLLama.Session do
  defstruct [
    seed: nil,
    model_name: nil,
    resource: nil
  ]

  # ⟦𓍓𓎣𓉤𓅭⟧ default_options :: auto-generated pointer for public function default_options
  def default_options(), do: ExLLama.Nif.__session_nif_default_session_options__()
  # ⟦𓅡𓊽𓁵𓈶⟧ advance_context_with_tokens :: auto-generated pointer for public function advance_context_with_tokens
  def advance_context_with_tokens(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context_with_tokens__(session.resource, context)
  # ⟦𓏂𓅒𓀌𓁣⟧ advance_context :: auto-generated pointer for public function advance_context
  def advance_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context__(session.resource, context)
  # ⟦𓁦𓍑𓎆𓁒⟧ start_completing_with :: auto-generated pointer for public function start_completing_with
  def start_completing_with(%__MODULE__{resource: _} = session, options) do
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
  # ⟦𓏬𓋂𓅃𓊛⟧ completion :: auto-generated pointer for public function completion
  def completion(%__MODULE__{resource: _} = session, max_tokens, stop), do: ExLLama.Nif.__session_nif_completion__(session.resource, max_tokens, stop)
  # ⟦𓐩𓁊𓏪𓎲⟧ model :: auto-generated pointer for public function model
  def model(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_model__(session.resource)
  # ⟦𓁨𓄇𓇯𓃷⟧ params :: auto-generated pointer for public function params
  def params(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_params__(session.resource)
  # ⟦𓊋𓊚𓆶𓈑⟧ context_size :: auto-generated pointer for public function context_size
  def context_size(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context_size__(session.resource)
  # ⟦𓃌𓍉𓍼𓊠⟧ context :: auto-generated pointer for public function context
  def context(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context__(session.resource)
  # ⟦𓆏𓃩𓍥𓃴⟧ truncate_context :: auto-generated pointer for public function truncate_context
  def truncate_context(%__MODULE__{resource: _} = session, n_tokens), do: ExLLama.Nif.__session_nif_truncate_context__(session.resource, n_tokens)
  # ⟦𓀿𓁨𓅄𓅺⟧ set_context_to_tokens :: auto-generated pointer for public function set_context_to_tokens
  def set_context_to_tokens(%__MODULE__{resource: _} = session, tokens), do: ExLLama.Nif.__session_nif_set_context_to_tokens__(session.resource, tokens)
  # ⟦𓈶𓁒𓉃𓈮⟧ set_context :: auto-generated pointer for public function set_context
  def set_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_set_context__(session.resource, context)
  # ⟦𓍛𓇖𓈜𓅉⟧ deep_copy :: auto-generated pointer for public function deep_copy
  def deep_copy(%__MODULE__{resource: _} = session) do
    with {:ok, copy} <- ExLLama.Nif.__session_deep_copy__(session.resource) do
      {:ok, put_in(copy, [Access.key(:model_name)], session.model_name)}
    end
  end

end
