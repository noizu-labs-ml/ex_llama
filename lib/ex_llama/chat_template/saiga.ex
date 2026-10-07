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
      # Handle system message
      {system_message, loop_messages} = case Enum.at(thread, 0) do
        %{role: :system, content: content} -> 
          {"#{bos_token}system\n#{String.trim(content)}#{eos_token}", Enum.drop(thread, 1)}
        _ -> 
          {"", thread}
      end

      # Process messages with validation
      lines = loop_messages
              |> Enum.with_index()
              |> Enum.map(
                   fn
                     {msg = %{role: role}, index} ->
                       # Validate alternation (user at even indices, assistant at odd)
                       expected_user = rem(index, 2) == 0
                       is_user = role == :user
                       
                       if is_user != expected_user && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, 
                           message: "Conversation roles must alternate user/bot/user/bot/...", 
                           handler: __MODULE__, 
                           entry: msg, 
                           row: index
                       end
                       
                       # Add system message before first message
                       prefix = if index == 0 && system_message != "", do: system_message, else: ""
                       
                       # Map assistant to bot for output
                       output_role = if role == :assistant, do: "bot", else: to_string(role)
                       prefix <> "#{bos_token}#{output_role}\n#{String.trim(msg.content)}#{eos_token}"
                   end
                 ) |> Enum.join("")
                 
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, lines <> "#{bos_token}bot\n"}
      else
        {:ok, lines}
      end
    end
  end
end
