defmodule ExLLama.SessionOptions do
  @moduledoc """
  Options used when creating an inference session (context).

  See `ExLLama.ContextParams` for the meaning of the numeric enum fields
  (`flash_attn_type`, `pooling_type`, `attention_type`, `type_k`, `type_v`, ...).
  Unset fields fall back to llama.cpp defaults — only explicitly provided keys
  are forwarded to the NIF.
  """

  defstruct [
    :seed,
    :n_ctx,
    :n_batch,
    :n_ubatch,
    :n_seq_max,
    :n_threads,
    :n_threads_batch,
    :flash_attn_type,
    :pooling_type,
    :attention_type,
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
    :op_offload,
    :kv_unified,
    :swa_full,
    :pooling,
  ]

  # ⟦𓉱𓋯𓍻𓉐⟧ new :: auto-generated pointer for public function new
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
