defmodule ExLLama.Model do
  defstruct [
    resource: nil,
    eos: nil,
    bos: nil,
    name: nil
  ]

  # <REMOVED UUID HERE> __eos__ :: auto-generated pointer for public function __eos__
  def __eos__(model) do
    {:ok, List.to_string(model.eos)}
  end

  # <REMOVED UUID HERE> __bos__ :: auto-generated pointer for public function __bos__
  def __bos__(model) do
    {:ok, List.to_string(model.bos)}
  end


  # <REMOVED UUID HERE> __model_name__ :: auto-generated pointer for public function __model_name__
  def __model_name__(model) do
    {:ok, model.name}
  end

  # <REMOVED UUID HERE> load_from_file :: auto-generated pointer for public function load_from_file
  def load_from_file(path), do: ExLLama.Nif.__model_nif_load_from_file__(path, ExLLama.ModelOptions.new())
  def load_from_file(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Nif.__model_nif_load_from_file__(path, opts)

  # <REMOVED UUID HERE> detokenize :: auto-generated pointer for public function detokenize
  def detokenize(model, token), do: ExLLama.Nif.__model_nif_detokenize__(model, token)

  # <REMOVED UUID HERE> token_to_byte_piece :: auto-generated pointer for public function token_to_byte_piece
  def token_to_byte_piece(model, token), do: ExLLama.Nif.__model_nif_token_to_byte_piece__(model, token)

  # <REMOVED UUID HERE> token_to_piece :: auto-generated pointer for public function token_to_piece
  def token_to_piece(model, token), do: ExLLama.Nif.__model_nif_token_to_piece__(model, token)

  # <REMOVED UUID HERE> decode_tokens :: auto-generated pointer for public function decode_tokens
  def decode_tokens(model, tokens), do: ExLLama.Nif.__model_nif_decode_tokens__(model, tokens)

  # <REMOVED UUID HERE> create_session :: auto-generated pointer for public function create_session
  def create_session(model) do
    with {:ok, options} <- ExLLama.Session.default_options do
        create_session(model, options)
    end
  end
  def create_session(model, options) do
    case ExLLama.Nif.__model_nif_create_session__(model, options) do
      {:ok, session} ->
        o = if is_struct(options), do: Map.from_struct(options), else: Map.new(options || [])
        if o[:seed], do: Process.put({:ex_llama_seed, session.resource}, o[:seed])
        Process.put({:ex_llama_model, session.resource}, model)
        if is_struct(options), do: Process.put({:ex_llama_params, session.resource}, options)
        {:ok, session}
      error -> error
    end
  end

  # <REMOVED UUID HERE> embeddings :: auto-generated pointer for public function embeddings
  def embeddings(model, inputs, options), do: ExLLama.Nif.__model_nif_embeddings__(model, inputs, options)

  # <REMOVED UUID HERE> bos :: auto-generated pointer for public function bos
  def bos(model), do: ExLLama.Nif.__model_nif_bos__(model)

  # <REMOVED UUID HERE> eos :: auto-generated pointer for public function eos
  def eos(model), do: ExLLama.Nif.__model_nif_eos__(model)

  # <REMOVED UUID HERE> nl :: auto-generated pointer for public function nl
  def nl(model), do: ExLLama.Nif.__model_nif_nl__(model)

  # <REMOVED UUID HERE> infill_prefix :: auto-generated pointer for public function infill_prefix
  def infill_prefix(model), do: ExLLama.Nif.__model_nif_infill_prefix__(model)

  # <REMOVED UUID HERE> infill_middle :: auto-generated pointer for public function infill_middle
  def infill_middle(model), do: ExLLama.Nif.__model_nif_infill_middle__(model)

  # <REMOVED UUID HERE> infill_suffix :: auto-generated pointer for public function infill_suffix
  def infill_suffix(model), do: ExLLama.Nif.__model_nif_infill_suffix__(model)

  # <REMOVED UUID HERE> eot :: auto-generated pointer for public function eot
  def eot(model), do: ExLLama.Nif.__model_nif_eot__(model)

  # <REMOVED UUID HERE> vocabulary_size :: auto-generated pointer for public function vocabulary_size
  def vocabulary_size(model), do: ExLLama.Nif.__model_nif_vocabulary_size__(model)

  # <REMOVED UUID HERE> embed_len :: auto-generated pointer for public function embed_len
  def embed_len(model), do: ExLLama.Nif.__model_nif_embed_len__(model)

  # <REMOVED UUID HERE> train_len :: auto-generated pointer for public function train_len
  def train_len(model), do: ExLLama.Nif.__model_nif_train_len__(model)
end
