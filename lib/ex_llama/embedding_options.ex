defmodule ExLLama.EmbeddingOptions do
  @moduledoc """
  Options for `ExLLama.Model.embeddings/3` embedding passes.

    * `n_threads` — threads for the embedding pass (default 1).
    * `n_threads_batch` — threads for the prompt/batch pass (default 1).

  Note: the current embeddings NIF bridge creates its embeddings-enabled
  context internally and does not yet forward these thread options — they are
  reserved API surface. Embedding behavior (pooling, causal masking) is tuned
  via the 0.4.0 `ExLLama.SessionOptions` / `ExLLama.ContextParams` fields
  `pooling_type` and `attention_type`.
  """
  defstruct [
    n_threads: 1,
    n_threads_batch: 1,
  ]
end
