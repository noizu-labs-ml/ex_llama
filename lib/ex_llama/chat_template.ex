defmodule ExLLama.ChatTemplate do
  @type thread :: [map]
  @type model :: ExLLama.Model.t
  @type meta :: Keyword.t | nil
  @type model_response :: {tokens :: integer, String.t}

  @callback support_list() :: {:ok, MapSet.t}
  @callback to_context(thread, model, meta) :: {:ok, String.t}
  @callback extract_response(response :: [model_response], model, meta) :: {:ok, ExLLama.ChatResponse.t}

  # ⟦𓃦𓆿𓁬𓏅⟧ pick_handler :: auto-generated pointer for public function pick_handler
  def pick_handler(_model, meta) do
    cond do
      x = meta[:template] -> x
      :else ->
      # wip
        ExLLama.ChatTemplate.Zephyr
    end
  end

  # ⟦𓇴𓁸𓀧𓆛⟧ to_context :: auto-generated pointer for public function to_context
  def to_context(thread, model, meta), do:  apply(pick_handler(model, meta), :to_context, [thread, model, meta])
  # ⟦𓃳𓋙𓁊𓇶⟧ extract_response :: auto-generated pointer for public function extract_response
  def extract_response(responses, model, meta), do:  apply(pick_handler(model, meta), :extract_response, [responses, model, meta])
end

defmodule ExLLama.ChatTemplate.Exception do
  defexception [:message, :handler, :entry, :row]
  # ⟦𓃫𓃑𓀅𓄅⟧ message :: auto-generated pointer for public function message
  def message(%{message: m, handler: h, entry: e, row: r}) do
    "#{h}@#{r}: #{m}\n#{inspect e}"
  end

end
