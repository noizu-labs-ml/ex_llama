defmodule ExLLama do
  use Rustler,
      otp_app: :ex_llama,
      crate: :erlang_llama_cpp_nif

  def load_model(path), do: __nif_load_model__(path, ExLLama.ModelOptions.new())
  def load_model(path, %ExLLama.ModelOptions{} = opts), do: __nif_load_model__(path, opts)
  def __nif_load_model__(_,_), do: :erlang.nif_error(:nif_not_loaded)

  def default_session_options(), do: __nif_default_session_options__()
  def __nif_default_session_options__(), do: :erlang.nif_error(:nif_not_loaded)

  def create_session(model), do: __nif_create_session__(model, default_session_options())
  def create_session(model, options), do: __nif_create_session__(model, options)
  def __nif_create_session__(_, _), do: :erlang.nif_error(:nif_not_loaded)

  def advance_context(session, content), do: __nif_advance_context__(session.resource, content)
  def __nif_advance_context__(_, _), do: :erlang.nif_error(:nif_not_loaded)

  def complete(session), do: __nif_complete__(session.resource, 1024)
  def __nif_complete__(_,_), do: :erlang.nif_error(:nif_not_loaded)

end
