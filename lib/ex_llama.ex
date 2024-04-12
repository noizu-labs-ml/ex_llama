defmodule ExLLama do
  def load_model(path), do: ExLLama.Nif.__context_nif_load_model__(path, ExLLama.ModelOptions.new())
  def load_model(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Nif.__context_nif_load_model__(path, opts)

  def default_session_options(), do: ExLLama.Nif.__context_nif_default_session_options__()

  def create_session(model), do: ExLLama.Nif.__context_nif_create_session__(model, default_session_options())
  def create_session(model, options), do: ExLLama.Nif.__context_nif_create_session__(model, options)

  def advance_context(session, content), do: ExLLama.Nif.__context_nif_advance_context__(session.resource, content)

  def complete(session), do: ExLLama.Nif.__context_nif_complete__(session.resource, 1024)
end
