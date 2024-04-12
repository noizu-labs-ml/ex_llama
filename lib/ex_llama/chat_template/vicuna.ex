defmodule ExLLama.ChatTemplate.Vicuna do
  @moduledoc """
  based on: https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/zephyr.jinja
  ```jinja
    {% if messages[0]['role'] == 'system' %}
      {% set loop_messages = messages[1:] %}
      {% set system_message = messages[0]['content'].strip() + '\n\n' %}
  {% else %}
      {% set loop_messages = messages %}
      {% set system_message = '' %}
  {% endif %}

  {{ bos_token + system_message }}
  {% for message in loop_messages %}
      {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
          {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
      {% endif %}

      {% if message['role'] == 'user' %}
          {{ 'USER: ' + message['content'].strip() + '\n' }}
      {% elif message['role'] == 'assistant' %}
          {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}
      {% endif %}

      {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
          {{ 'ASSISTANT:' }}
      {% endif %}
  {% endfor %}
  ````
  """


  def support_list() do
    []
  end

  defp format_line(message = %{role: :system}, eos_token) do
    "#{String.trim(message.content)}\n\n"
  end
  defp format_line(message, eos_token) do
    "#{message.role |> String.upcase()}: #{String.trim(message.content)}#{eos_token}\n"
  end

  def extract_response(responses, model, options) do
    with {:ok, model_name} <- ExLLama.Model.__model_name__(model),
         {:ok, eos_token} <- ExLLama.Model.__eos__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         x = x
                             |> String.trim()
                             |> String.trim_trailing(eos_token)
                         finish_reason = if (tokens < options[:max_tokens]), do: :stop, else: :max_tokens
                         %ExLLama.ChatCompletion.Choice{index: index, message: x, finish_reason: finish_reason}
                     end)
      completion_tokens = Enum.map(responses, fn {tokens,_} -> tokens end) |> Enum.max()
      prompt_tokens = options[:prompt_tokens]
      usage = %ExLLama.ChatCompletion.Usage{prompt_tokens: prompt_tokens, total_tokens: completion_tokens + prompt_tokens, completion_tokens: completion_tokens}
      completion = %ExLLama.ChatCompletion{id: nil, model: model_name, seed: options[:seed], choices: choices, usage: usage}
      {:ok, completion}
    end
  end

  def to_context(thread, model, options) do
    with {:ok, eos_token} <- ExLLama.Model.__eos__(model),
         {:ok, bos_token} <- ExLLama.Model.__bos__(model) do
      system_message_offset =  if (Enum.at(thread, 0)[:role] == :system), do: 1, else: 0
      lines = thread
              |> Enum.with_index()
              |> Enum.map(
                   fn
                     {msg = %{role: :system = role, content: content}, 0} ->
                       format_line(msg, eos_token)
                     {msg = %{role: :system = role, content: content}, index} ->
                       unless options[:strict] == false do
                         raise ExLLama.ChatTemplate.Exception, message: "Only the first message may be from system. Use a different handler or pass `strict: false` to allow", handler: __MODULE__, entry: msg, row: index
                       end
                       format_line(msg, eos_token)

                     {msg =%{role: :assistant = role, content: content}, index} ->
                       unless options[:strict] == false or index <= (1 + system_message_offset) do
                         if Enum.at(thread, index - 2)[:role] != role do
                           raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                         end
                       end
                       format_line(msg, eos_token)

                     {msg = %{role: :user = role, content: content}, index} ->
                       unless options[:strict] == false or index <= (2 + system_message_offset) do
                         if Enum.at(thread, index - 2)[:role] != role do
                           raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                         end
                       end
                       format_line(msg, eos_token)

                     {msg = %{role: role, content: content}, index} ->
                       unless options[:strict] == false or options[:expanded_roles] do
                         raise ExLLama.ChatTemplate.Exception, message: "Only the first user,assistant,system roles are supported. Use a different handler or pass `strict: false` to allow", handler: __MODULE__, entry: msg, row: index
                       end
                       format_line(msg, eos_token)
                   end
                 ) |> Enum.join("\n")
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, bos_token <> lines <> "ASSISTANT:"}
      else
        {:ok, bos_token <> lines}
      end
    end
  end

end
