defmodule ExLLama.ChatTemplate.ChatQA do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/chatqa.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
      {% set system_message = 'System: ' + messages[0]['content'] | trim %}
      {% set messages = messages[1:] %}
  {% else %}
      {% set system_message = '' %}
  {% endif %}

  {% if messages[0]['role'] == 'context' %}
      {% set context_message = '\n\n' + messages[0]['content'] | trim %}
      {% set messages = messages[1:] %}
  {% else %}
      {% set context_message = '' %}
  {% endif %}

  {{ bos_token + system_message + context_message}}
  {% for message in messages %}
      {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
          {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
      {% endif %}

      {% if message['role'] == 'user' %}
          {{ '\n\nUser: ' + message['content'] | trim }}
      {% elif message['role'] == 'assistant' %}
          {{ '\n\nAssistant: '  + message['content'] | trim }}
      {% endif %}
  {% endfor %}

  {% if add_generation_prompt %}
      {{ '\n\nAssistant:' }}
  {% endif %}
  ```
  """

  def support_list() do
    [{~r"^.*chatqa.*$", 1}]
  end

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

  def to_context(thread, model, options) do
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model) do
      # Handle system message
      {system_message, remaining_messages} = case Enum.at(thread, 0) do
        %{role: :system, content: content} -> 
          {"System: #{String.trim(content)}", Enum.drop(thread, 1)}
        _ -> 
          {"", thread}
      end

      # Handle context message
      {context_message, loop_messages} = case Enum.at(remaining_messages, 0) do
        %{role: :context, content: content} -> 
          {"\n\n#{String.trim(content)}", Enum.drop(remaining_messages, 1)}
        _ -> 
          {"", remaining_messages}
      end

      # Build the message thread
      lines = loop_messages
              |> Enum.with_index()
              |> Enum.map(
                   fn
                     {msg = %{role: :user}, index} ->
                       if rem(index, 2) != 0 && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                       "\n\nUser: #{String.trim(msg.content)}"

                     {msg = %{role: :assistant}, index} ->
                       if rem(index, 2) != 1 && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                       "\n\nAssistant: #{String.trim(msg.content)}"

                     {msg, index} ->
                       unless options[:strict] == false or options[:expanded_roles] do
                         raise ExLLama.ChatTemplate.Exception, message: "Only user, assistant, system, and context roles are supported", handler: __MODULE__, entry: msg, row: index
                       end
                       ""
                   end
                 ) |> Enum.join("")
      
      result = bos_token <> system_message <> context_message <> lines
      
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
        {:ok, result <> "\n\nAssistant:"}
      else
        {:ok, result}
      end
    end
  end
end