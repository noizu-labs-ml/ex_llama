defmodule ExLLama.ContextParams do
  @moduledoc """
  Parameters for a llama.cpp inference context.

  Numeric enum fields map 1:1 to their llama.cpp counterparts:

    * `flash_attn_type` - `-1` auto, `0` disabled, `1` enabled
    * `pooling_type` - `-1` unspecified, `0` none, `1` mean, `2` cls, `3` last, `4` rank
    * `attention_type` - `-1` unspecified, `0` causal, `1` non-causal (embeddings)
    * `rope_scaling_type` - `0` none, `1` linear, `2` yarn (llama.cpp `LLAMA_ROPE_SCALING_*`)
    * `type_k` / `type_v` - KV cache ggml type: `1` F16 (default), `8` Q8_0, `2` Q4_0, `30` BF16
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
    :pooling
  ]

  @type t :: %__MODULE__{
               seed: non_neg_integer(),
               n_ctx: non_neg_integer(),
               n_batch: non_neg_integer(),
               n_ubatch: non_neg_integer(),
               n_seq_max: non_neg_integer(),
               n_threads: non_neg_integer(),
               n_threads_batch: non_neg_integer(),
               flash_attn_type: integer(),
               pooling_type: integer(),
               attention_type: integer(),
               rope_scaling_type: integer(),
               rope_freq_base: float(),
               rope_freq_scale: float(),
               yarn_ext_factor: float(),
               yarn_attn_factor: float(),
               yarn_beta_fast: float(),
               yarn_beta_slow: float(),
               yarn_orig_ctx: non_neg_integer(),
               type_k: non_neg_integer(),
               type_v: non_neg_integer(),
               embedding: boolean(),
               offload_kqv: boolean(),
               op_offload: boolean(),
               kv_unified: boolean(),
               swa_full: boolean(),
               pooling: boolean()
             }
end
