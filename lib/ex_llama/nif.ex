defmodule ExLLama.Nif do
  use Rustler,
      otp_app: :ex_llama,
      crate: :erlang_llama_cpp_nif

  defstruct [
    resource: nil
  ]

  def __model_nif_load_from_file__(_,_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_detokenize__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_token_to_byte_piece__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_token_to_piece__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_decode_tokens__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_create_session__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_embeddings__(_, _, _), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_bos__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_eos__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_nl__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_infill_prefix__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_infill_middle__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_infill_suffix__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_eot__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_vocabulary_size__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_embed_len__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __model_nif_train_len__(_), do: :erlang.nif_error(:nif_not_loaded)

  def __context_nif_load_model__(_,_), do: :erlang.nif_error(:nif_not_loaded)
  def __context_nif_default_session_options__(), do: :erlang.nif_error(:nif_not_loaded)
  def __context_nif_create_session__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __context_nif_advance_context__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __context_nif_complete__(_,_), do: :erlang.nif_error(:nif_not_loaded)


  def __session_nif_default_session_options__(), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_advance_context_with_tokens__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_advance_context__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_completion__(_, _, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_model__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_params__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_context_size__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_context__(_), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_truncate_context__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_set_context_to_tokens__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_nif_set_context__(_, _), do: :erlang.nif_error(:nif_not_loaded)
  def __session_deep_copy__(_), do: :erlang.nif_error(:nif_not_loaded)






end
