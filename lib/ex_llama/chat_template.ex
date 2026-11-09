defmodule ExLLama.ChatTemplate do
  @moduledoc """
  Chat template behaviour and dispatcher behind `ExLLama.chat_completion/3`.

  A template renders a message thread (list of `%{role: _, content: _}` maps)
  into model-ready prompt text and extracts a structured completion from raw
  generation results.

  ## Behaviour callbacks

    * `support_list/0` — `{:ok, map_set}` of model names the template matches.
    * `to_context/3` — `{:ok, prompt}` for a thread (bos/eos-aware).
    * `extract_response/3` — `{:ok, %GenAI.ChatCompletion{}}` from
      `{token_count, text}` generation pairs.

  Handler selection (`pick_handler/2`): an explicit `template:` option module
  wins; otherwise `ExLLama.ChatTemplate.Zephyr` is used.

  18 built-in templates ship under `ExLLama.ChatTemplate.*` — Alpaca,
  AmberChat, ChatML, ChatQA, FalconInstruct, GemmaInstruct, GraniteInstruct,
  LLama2Chat, Llama3Instruct, MistralInstruct, OpenChat, Phi3, Phi3Small,
  QwenInstruct, Saiga, SolarInstruct, Vicuna, Zephyr. Implement the behaviour
  in your own module and pass it via `template:` to plug in custom formats.

  `ExLLama.ChatTemplate.Exception` is raised when template rendering fails;
  it carries the handler, entry and failing row for diagnosis.
  """
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
