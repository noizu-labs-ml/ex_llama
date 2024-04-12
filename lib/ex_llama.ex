defmodule ExLLama do
  def load_model(path), do: ExLLama.Model.load_from_file(path)
  def load_model(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Model.load_from_file(path, opts)

  def create_session(model), do: ExLLama.Model.create_session(model)
  def create_session(model, options), do: ExLLama.Model.create_session(model, options)

  def advance_context(session, content), do: ExLLama.Session.advance_context(session, content)
  def completion(session, max_tokens, stop), do: ExLLama.Session.completion(session, max_tokens, stop)


  @default_choices 1
  @default_max_tokens 512

  def chat_completion(model, thread, options) do

    so = cond do
      x = options[:session_options] -> put_in(x, [:seed], options[:seed])
      x = options[:seed] -> [seed: x]
      :else -> nil
    end
    options = update_in(options || [], [:add_generation_prompt],
      fn
        x when is_nil(x) -> true
        x -> x
      end
    )
    session_options = ExLLama.SessionOptions.new(so)
    seed = session_options.seed
    choices = options[:choices] || @default_choices
    max_tokens = options[:max_tokens] || @default_max_tokens
    with {:ok, session} <- ExLLama.create_session(model, session_options),
         {:ok, thread_context} <- ExLLama.ChatTemplate.to_context(thread, model, options),
         {:ok, _} <- ExLLama.Session.set_context(session, thread_context),
         {:ok, prompt_tokens} = ExLLama.Session.context_size(session) do
      choices = Enum.map(1..choices,
                  fn(_) ->
                    with {:ok, %{content: result, token_length: l}} <- ExLLama.Session.completion(session, max_tokens, nil) do
                      {:ok, {l, result}}
                    end
                  end
                )
                |> Enum.filter(
                     fn
                       {:ok, _} -> true
                       _ -> false
                     end)
                |> Enum.map(fn {:ok, x} -> x end)

      options = (options || [])
                |> put_in([:prompt_tokens], prompt_tokens)
      ExLLama.ChatTemplate.extract_response(choices, model, options)
    end
  end
end
