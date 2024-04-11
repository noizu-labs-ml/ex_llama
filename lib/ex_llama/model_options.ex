defmodule ExLLama.ModelOptions do
  defstruct [
    :n_gpu_layers,
    :split_mode,
    :main_gpu,
    :vocab_only,
    :use_mmap,
    :use_mlock
  ]

  @type t :: %__MODULE__{
               n_gpu_layers: non_neg_integer(),
               split_mode: String.t, #  :none | :layer | :row,
               main_gpu: non_neg_integer(),
               vocab_only: boolean(),
               use_mmap: boolean(),
               use_mlock: boolean()
             }

  def new() do
    %__MODULE__{
      n_gpu_layers: 0,
      split_mode: "none",
      main_gpu: 0,
      vocab_only: false,
      use_mmap: false,
      use_mlock: false
    }
  end
end
