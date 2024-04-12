defmodule ExLLama.ChatTemplate.OpenChat do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/openchat.jinja]]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '<|end_of_turn|>' %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {{ bos_token + system_message }}
  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>' }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ 'GPT4 Correct Assistant:' }}
    {% endif %}
  {% endfor %}
  ````
  """


  def support_list() do
    [ ]
  end


  defp format_line(%{role: :system} = message) do
    "#{String.trim(message.content)}<|end_of_turn|>"
  end
  defp format_line(message) do
    "GPT4 Correct #{String.capitalize(to_string(message.role))}: #{String.trim(message.content)}<|end_of_turn|>"
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
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model) do
      lines = thread
              |> Enum.map(&format_line/1)
              |> Enum.join("")
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, bos_token <> lines <> "GPT4 Correct Assistant:"}
      else
        {:ok, bos_token <> lines}
      end
    end
  end
end
