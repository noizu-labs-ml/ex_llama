defmodule ExLLama.Session do
  defstruct [
    resource: nil
  ]




  def advance_context_with_tokens(session, context), do: ExLLama.Nif.__session_nif_advance_context_with_tokens__(session.resource, context)
  def advance_context(session, context), do: ExLLama.Nif.__session_nif_advance_context__(session.resource, context)
  def completion(session, max_predictions), do: ExLLama.Nif.__session_nif_completion__(session.resource, max_predictions)
  def model(session), do: ExLLama.Nif.__session_nif_model__(session.resource)
  def params(session), do: ExLLama.Nif.__session_nif_params__(session.resource)
  def context_size(session), do: ExLLama.Nif.__session_nif_context_size__(session.resource)
  def context(session), do: ExLLama.Nif.__session_nif_context__(session.resource)
  def truncate_context(session, n_tokens), do: ExLLama.Nif.__session_nif_truncate_context__(session.resource, n_tokens)
  def set_context_to_tokens(session, tokens), do: ExLLama.Nif.__session_nif_set_context_to_tokens__(session.resource, tokens)
  def set_context(session, context), do: ExLLama.Nif.__session_nif_set_context__(session.resource, context)
  def deep_copy(session, context), do: ExLLama.Nif.__session_deep_copy__(session.resource, context)

end
