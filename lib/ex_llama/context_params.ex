defmodule ExLLama.ContextParams do
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
    :pooling
  ]

  @type t :: %__MODULE__{
               seed: non_neg_integer(),
               n_ctx: non_neg_integer(),
               n_batch: non_neg_integer(),
               n_threads: non_neg_integer(),
               n_threads_batch: non_neg_integer(),
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
               pooling: boolean()
             }
end
