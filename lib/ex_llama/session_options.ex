defmodule ExLLama.SessionOptions do
  defstruct [
    :seed,
    :n_ctx,
    :n_batch,
    :n_threads,
    :n_threads_batch,
    :rope_scaling_type,
    :rope_freq_base,
    :rope_freq_scale,
    :yarn_ext_factor,
    :yarn_attn_factor,
    :yarn_beta_fast,
    :yarn_beta_slow,
    :yarn_orig_ctx,
    :type_k,
    :type_v,
    :embedding,
    :offload_kqv,
    :pooling,
  ]

  def new() do
    {:ok, session_options} = ExLLama.Session.default_options()
    session_options
  end
  def new(nil), do: new()
  def new(%__MODULE__{} = x), do: x
  def new(params) when is_list(params), do: new(Map.new(params))
  def new(params) when is_map(params) do
    {:ok, session_options} = ExLLama.Session.default_options()
    so = Map.from_struct(session_options)
    allowed_keys = Map.keys(so)
    po = Map.take(params, allowed_keys)
    unless po == %{} do
      ExLLama.SessionOptions.__struct__(Map.merge(so, po))
    else
      session_options
    end
  end

end
