defmodule ExLLama.ChatTemplate do
  @callback to_context(thread :: [map]) :: {:ok, String.t}
  @callback extract_response(thread :: [map]) :: {:ok, term}

  def to_context(handler, thread), do:  apply(handler, :to_context, [thread])
  def extract_response(handler, responses), do:  apply(handler, :extract_response, [responses])

  defmodule TinyLlama do
    @behaviour ExLLama.ChatTemplate

    def to_context(thread) do
      str = Enum.map(thread,
        fn
          %{role: role, content: content} -> "<|#{role}|>\n #{content}</s>"
        end
      ) |> Enum.join("\n")
      {:ok, str <> "\n<|assistant|>\n"}
    end

    def extract_response(response) do
      choices = Enum.map(response, fn
       " " <> x ->
         x = String.trim_trailing(x)
         if String.ends_with?(x, "</s>") do
           x = String.trim_trailing(x, "</s>")
           %{role: "assistant", content: x, reason: :end}
         else
           %{role: "assistant", content: x, reason: :tokens}
         end

        x ->
          x = String.trim_trailing(x)
          if String.ends_with?(x, "</s>") do
            x = String.trim_trailing(x, "</s>")
            %{role: "assistant", content: x, reason: :end}
          else
            %{role: "assistant", content: x, reason: :tokens}
          end
      end)
      {:ok, %{choices: choices}}
    end
  end
end

defmodule ExLLama do
  def load_model(path), do: ExLLama.Model.load_from_file(path)
  def load_model(path, %ExLLama.ModelOptions{} = opts), do: ExLLama.Model.load_from_file(path, opts)

  def create_session(model), do: ExLLama.Model.create_session(model)
  def create_session(model, options), do: ExLLama.Model.create_session(model, options)

  def advance_context(session, content), do: ExLLama.Session.advance_context(session, content)
  def completion(session, max_tokens, stop), do: ExLLama.Session.completion(session, max_tokens, stop)

  def chat_completion(model, thread, options \\ nil) do
    max_tokens = options[:max_tokens] || 512
    batch = options[:batch] || 3
    template = options[:template] || ExLLama.ChatTemplate.TinyLlama
    # todo setup session opts
    {:ok, session_options} = ExLLama.Session.default_options()
    session_options = session_options
                      |> then(fn opts -> options[:seed] && Map.put(opts, :seed, options[:seed]) || opts  end)
    # other args
    {:ok, session} = ExLLama.create_session(model, session_options)
    {:ok, thread_context} = ExLLama.ChatTemplate.to_context(template , thread)
    ExLLama.Session.set_context(session, thread_context)
    response = Enum.map(1..batch, fn(_) ->
      {:ok, result} = ExLLama.Session.completion(session, max_tokens, "^ignoreme") # todo pass nil to nif to bypass regex check
      result
    end)
    ExLLama.ChatTemplate.extract_response(template, response)
  end




end
