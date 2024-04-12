defmodule ExLLama.Session do
  defstruct [
    resource: nil
  ]

  def default_options(), do: ExLLama.Nif.__session_nif_default_session_options__()
  def advance_context_with_tokens(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context_with_tokens__(session.resource, context)
  def advance_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_advance_context__(session.resource, context)
  def start_completing_with(%__MODULE__{resource: _} = session, options) do
    # @TODO this is a little hacky, threading should be done in nif but passing env into the thread is unsupported.
    max_tokens = options[:max_tokens] || 512
    pid = with nil <- options[:pid] do
      self()
    end
    spawn fn ->
      o = ExLLama.Nif.__session_nif_start_completing_with__(pid, session.resource, max_tokens)
      send(pid, o)
    end
    :ok
  end
  def completion(%__MODULE__{resource: _} = session, max_tokens, stop), do: ExLLama.Nif.__session_nif_completion__(session.resource, max_tokens, stop)
  def model(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_model__(session.resource)
  def params(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_params__(session.resource)
  def context_size(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context_size__(session.resource)
  def context(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_nif_context__(session.resource)
  def truncate_context(%__MODULE__{resource: _} = session, n_tokens), do: ExLLama.Nif.__session_nif_truncate_context__(session.resource, n_tokens)
  def set_context_to_tokens(%__MODULE__{resource: _} = session, tokens), do: ExLLama.Nif.__session_nif_set_context_to_tokens__(session.resource, tokens)
  def set_context(%__MODULE__{resource: _} = session, context), do: ExLLama.Nif.__session_nif_set_context__(session.resource, context)
  def deep_copy(%__MODULE__{resource: _} = session), do: ExLLama.Nif.__session_deep_copy__(session.resource)

end
