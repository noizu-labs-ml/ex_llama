defmodule ExLLama.Model do
  defstruct [
    resource: nil
  ]

  def load_from_file(path), do: ExLLama.Nif.__model_nif_load_from_file__(path, ExLLama.ModelOptions.new())
  def load_from_file(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Nif.__model_nif_load_from_file__(path, opts)
  def detoknize(model, token), do: ExLLama.Nif.__model_nif_detokenize__(model, token)
  def token_to_byte_piece(model, token), do: ExLLama.Nif.__model_nif_token_to_byte_piece__(model, token)
  def token_to_piece(model, token), do: ExLLama.Nif.__model_nif_token_to_piece__(model, token)
  def decode_tokens(model, tokens), do: ExLLama.Nif.__model_nif_decode_tokens__(model, tokens)
  def create_session(model, options), do: ExLLama.Nif.__model_nif_create_session__(model, options)
  def embeddings(model, inputs, options), do: ExLLama.Nif.__model_nif_embeddings__(model, inputs, options)
  def bos(model), do: ExLLama.Nif.__model_nif_bos__(model)
  def eos(model), do: ExLLama.Nif.__model_nif_eos__(model)
  def nl(model), do: ExLLama.Nif.__model_nif_nl__(model)
  def infill_prefix(model), do: ExLLama.Nif.__model_nif_infill_prefix__(model)
  def infill_middle(model), do: ExLLama.Nif.__model_nif_infill_middle__(model)
  def infill_suffix(model), do: ExLLama.Nif.__model_nif_infill_suffix__(model)
  def eot(model), do: ExLLama.Nif.__model_nif_eot__(model)
  def vocabulary_size(model), do: ExLLama.Nif.__model_nif_vocabulary_size__(model)
  def embed_len(model), do: ExLLama.Nif.__model_nif_embed_len__(model)
  def train_len(model), do: ExLLama.Nif.__model_nif_train_len__(model)

end
