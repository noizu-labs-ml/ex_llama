defmodule ExLLama.ChatTemplate.GemmaInstruct do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/gemma-it.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}

    {% if (message['role'] == 'assistant') %}
        {% set role = 'model' %}
    {% else %}
        {% set role = message['role'] %}
    {% endif %}

    {{ '<start_of_turn>' + role + '\n' + content.strip() + '<end_of_turn>\n' }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{'<start_of_turn>model\n'}}
    {% endif %}
  {% endfor %}
  ````
  """


  def support_list() do
    [ ]
  end

  defp format_line(message, eos_token) do
    role = case message.role do
      :assistant -> :model
      x -> x
    end
    "<start_of_turn>#{role}\n#{String.trim(message.content)}<end_of_turn>\n"
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

  def compact(thread, acc \\ [])
  def compact([], acc), do: acc
  def compact([h], acc), do: [h | acc]
  def compact([%{role: :system} = s, %{role: :user} = u|t], acc) do
    c = %{u| content: "#{String.trim(s.content)}\n\n#{String.trim(u.content)}"}
    compact(t, [c| acc])
  end
  def compact([h|t], acc), do: compact(t, [h|acc])

  def to_context(thread, model, options) do
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model),
         {:ok, eos_token} <- ExLLama.Model.__eos__(model) do
      lines = thread
              |> compact()
              |> Enum.reverse()
              |> Enum.map(&format_line(&1, eos_token))
              |> Enum.join("")
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, lines <> "<start_of_turn>model\n"}
      else
        {:ok, lines}
      end
    end
  end

end
