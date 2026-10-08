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

  ## Deprecated: use_mmap / use_mlock

  Kept as deprecated struct fields for backward compatibility with the pre-0.4
  API. When set, they are translated to `load_mode` (`"mmap"`, `"mlock"`,
  `"mmap+mlock"`, or `"none"` when both are false). An explicit `:load_mode`
  always wins. They are never forwarded to the NIF directly.
  """

  require Logger

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
    :load_mtp,
    # Deprecated: translated to :load_mode in new/1 and in the load bridge.
    :use_mmap,
    :use_mlock
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
               load_mtp: boolean(),
               use_mmap: boolean() | nil,
               use_mlock: boolean() | nil
             }

  # <REMOVED UUID HERE> new :: auto-generated pointer for public function new
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
    params = translate_deprecated(params)
    base = Map.from_struct(new())
    allowed_keys = Map.keys(base) ++ [:use_mmap, :use_mlock]
    __struct__(Map.merge(base, Map.take(params, allowed_keys)))
  end

  # Resolves the effective load_mode for a loaded-options map or struct,
  # applying the deprecated use_mmap/use_mlock translation. An explicit
  # non-nil :load_mode on the caller's params wins over the derived value.
  @doc false
  def resolve_load_mode(params) when is_map(params) do
    mmap = Map.get(params, :use_mmap)
    mlock = Map.get(params, :use_mlock)

    if mmap == nil and mlock == nil do
      Map.get(params, :load_mode)
    else
      Logger.warning(
        "ExLLama.ModelOptions: :use_mmap/:use_mlock are deprecated; " <>
          "translated to :load_mode (see moduledoc). Set :load_mode directly."
      )

      derived =
        cond do
          mmap -> if mlock, do: "mmap+mlock", else: "mmap"
          mlock -> "mlock"
          true -> "none"
        end

      case Map.get(params, :load_mode) do
        nil -> derived
        # "auto" is the struct default, not an explicit override, when the
        # caller also set the deprecated booleans.
        "auto" -> derived
        explicit -> explicit
      end
    end
  end

  defp translate_deprecated(params) do
    if Map.has_key?(params, :use_mmap) or Map.has_key?(params, :use_mlock) do
      Map.put(params, :load_mode, resolve_load_mode(params))
    else
      params
    end
  end
end
