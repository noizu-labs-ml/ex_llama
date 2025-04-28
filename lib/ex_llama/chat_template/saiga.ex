defmodule ExLLama.ChatTemplate.Saiga do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/saiga.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = bos_token + 'system' + '\n' + messages[0]['content'].strip() + eos_token %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/bot/user/bot/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ system_message }}
    {% endif %}

    {{ bos_token + message['role'] + '\n' + message['content'].strip() + eos_token }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ bos_token + 'bot\n' }}
    {% endif %}
  {% endfor %}
  ````
  """

  def support_list() do
    [ ]
  end

  defp format_line(message, bos_token, eos_token) do
    "#{bos_token}#{message.role}\n#{String.trim(message.content)}#{eos_token}"
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
                         x = GenAI.Message.assistant(x)
                         finish_reason = if (tokens < options[:max_tokens]), do: :stop, else: :max_tokens
                         %GenAI.ChatCompletion.Choice{index: index, message: x, finish_reason: finish_reason}
                     end)
      completion_tokens = Enum.map(responses, fn {tokens,_} -> tokens end) |> Enum.max()
      prompt_tokens = options[:prompt_tokens]
      usage = %GenAI.ChatCompletion.Usage{prompt_tokens: prompt_tokens, total_tokens: completion_tokens + prompt_tokens, completion_tokens: completion_tokens}
      completion = %GenAI.ChatCompletion{id: nil, model: model_name, seed: options[:seed], choices: choices, usage: usage}
      {:ok, completion}
    end
  end

  def to_context(thread, model, options) do
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model),
         {:ok, eos_token} <- ExLLama.Model.__eos__(model) do
      lines = thread
              |> Enum.map(&format_line(&1, bos_token, eos_token))
              |> Enum.join("")
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, lines <> "#{bos_token}bot\n"}
      else
        {:ok, lines}
      end
    end
  end
end
