defmodule ExLLama do
  def load_model(path), do: ExLLama.Model.load_from_file(path)
  def load_model(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Model.load_from_file(path, opts)

  def create_session(model), do: ExLLama.Model.create_session(model)
  def create_session(model, options), do: ExLLama.Model.create_session(model, options)

  def advance_context(session, content), do: ExLLama.Session.advance_context(session, content)
  def completion(session, max_tokens, stop), do: ExLLama.Session.completion(session, max_tokens, stop)
end
