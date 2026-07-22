defmodule ExLLama.ChatTemplate.AmberChat do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/amberchat.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n' %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ bos_token + system_message }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '###Human: ' + message['content'].strip() + '\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '###Assistant: ' + message['content'].strip() + '\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '###Assistant:' }}
    {% endif %}
  {% endfor %}
  ````
  """

  # ⟦𓈲𓏟𓄶𓅔⟧ support_list :: auto-generated pointer for public function support_list
  def support_list() do
    [{~r"^amber.*$", 1}]
  end

  defp format_message(message) do
    case message.role do
      :user -> "###Human: #{String.trim(message.content)}\n"
      :assistant -> "###Assistant: #{String.trim(message.content)}\n"
      _ -> ""
    end
  end

  # ⟦𓁧𓋐𓈄𓈽⟧ extract_response :: auto-generated pointer for public function extract_response
  def extract_response(responses, model, options) do
    with {:ok, model_name} <- ExLLama.Model.__model_name__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         content = x |> String.trim()
                         message = GenAI.Message.assistant(content)
                         finish_reason = if (tokens < options[:max_tokens]), do: :stop, else: :max_tokens
                         %GenAI.ChatCompletion.Choice{index: index, message: message, finish_reason: finish_reason}
                     end)
      completion_tokens = Enum.map(responses, fn {tokens,_} -> tokens end) |> Enum.max()
      prompt_tokens = options[:prompt_tokens]
      usage = %GenAI.ChatCompletion.Usage{prompt_tokens: prompt_tokens, total_tokens: completion_tokens + prompt_tokens, completion_tokens: completion_tokens}
      completion = %GenAI.ChatCompletion{id: nil, model: model_name, seed: options[:seed], choices: choices, usage: usage}
      {:ok, completion}
    end
  end

  # ⟦𓄲𓅹𓂑𓇼⟧ to_context :: auto-generated pointer for public function to_context
  def to_context(thread, model, options) do
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model) do
      {system_message, loop_messages} = case Enum.at(thread, 0) do
        %{role: :system, content: content} -> 
          {String.trim(content) <> "\n", Enum.drop(thread, 1)}
        _ -> 
          {"", thread}
      end

      lines = loop_messages
              |> Enum.with_index()
              |> Enum.map(
                   fn
                     {msg = %{role: :user}, index} ->
                       if rem(index, 2) != 0 && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                       prefix = if index == 0, do: bos_token <> system_message, else: ""
                       prefix <> format_message(msg)

                     {msg = %{role: :assistant}, index} ->
                       if rem(index, 2) != 1 && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                       format_message(msg)

                     {msg, index} ->
                       unless options[:strict] == false or options[:expanded_roles] do
                         raise ExLLama.ChatTemplate.Exception, message: "Only user, assistant, and system roles are supported", handler: __MODULE__, entry: msg, row: index
                       end
                       format_message(msg)
                   end
                 ) |> Enum.join("")
      
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, lines <> "###Assistant:"}
      else
        {:ok, lines}
      end
    end
  end
end
