defmodule ExLLama.ChatTemplate do
  @type thread :: [map]
  @type model :: ExLLama.Model.t
  @type meta :: Keyword.t | nil
  @type model_response :: {tokens :: integer, String.t}

  @callback support_list() :: {:ok, MapSet.t}
  @callback to_context(thread, model, meta) :: {:ok, String.t}
  @callback extract_response(response :: [model_response], model, meta) :: {:ok, ExLLama.ChatResponse.t}

  def pick_handler(model, meta) do
    cond do
      x = meta[:template] -> x
      :else ->
      # wip
        ExLLama.ChatTemplate.Zephyr
    end
  end

  def to_context(thread, model, meta), do:  apply(pick_handler(model, meta), :to_context, [thread, model, meta])
  def extract_response(responses, model, meta), do:  apply(pick_handler(model, meta), :extract_response, [responses, model, meta])
end

defmodule ExLLama.ChatTemplate.Exception do
  defexception [:message, :handler, :entry, :row]
  def message(%{message: m, handler: h, entry: e, row: r}) do
    "#{h}@#{r}: #{m}\n#{inspect e}"
  end

end
