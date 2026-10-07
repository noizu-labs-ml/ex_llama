defmodule ExLLama.ModelOptions do
  @moduledoc """
  Options for loading a GGUF model via llama.cpp.

  ## load_mode

  llama.cpp v0.6.0 replaced the old `use_mmap`/`use_mlock` booleans with a single
  `load_mode`:

    * `"auto"` - auto-detect based on device capabilities (default)
    * `"none"` - plain read, no special loading mode
    * `"mmap"` - memory map the model
    * `"mlock"` - force the model to stay in RAM
    * `"mmap+mlock"` - mmap + keep in RAM (`"mmap_mlock"` also accepted)
    * `"direct_io"` / `"dio"` - direct I/O if available

  ## split_mode / use_extra_bufts

  Left `nil` by default so the llama.cpp defaults (`"layer"` split, extra
  buffer types enabled) are preserved; set them explicitly to override.

  ## lazy_mode

    * `0` - off: read whole tensors up front (default)
    * `1` - auto: lazy only for marked tensors larger than 4 GiB (requires mmap)
    * `2` - on: read rows of marked tensors on demand (requires mmap)
  """

  defstruct [
    :n_gpu_layers,
    :split_mode,
    :main_gpu,
    :load_mode,
    :lazy_mode,
    :vocab_only,
    :check_tensors,
    :use_extra_bufts,
    :no_host,
    :load_mtp
  ]

  @type t :: %__MODULE__{
               n_gpu_layers: non_neg_integer(),
               split_mode: String.t, #  "none" | "layer" | "row" | "tensor"
               main_gpu: non_neg_integer(),
               load_mode: String.t,  #  "auto" | "none" | "mmap" | "mlock" | "mmap+mlock" | "direct_io"
               lazy_mode: 0 | 1 | 2,
               vocab_only: boolean(),
               check_tensors: boolean(),
               use_extra_bufts: boolean(),
               no_host: boolean(),
               load_mtp: boolean()
             }

  # ⟦𓄊𓄆𓈎𓅉⟧ new :: auto-generated pointer for public function new
  def new() do
    %__MODULE__{
      n_gpu_layers: 0,
      main_gpu: 0,
      load_mode: "auto",
      lazy_mode: 0,
      vocab_only: false,
      check_tensors: false,
      no_host: false,
      load_mtp: false
    }
  end

  def new(nil), do: new()
  def new(%__MODULE__{} = x), do: x
  def new(params) when is_list(params), do: new(Map.new(params))

  def new(params) when is_map(params) do
    base = Map.from_struct(new())
    allowed_keys = Map.keys(base)
    __struct__(Map.merge(base, Map.take(params, allowed_keys)))
  end
end
