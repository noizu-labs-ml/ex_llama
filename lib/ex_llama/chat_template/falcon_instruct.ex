defmodule ExLLama.ChatTemplate.FalconInstruct do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/falcon-instruct.jinja]]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
  {% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
  {% endif %}

  {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ system_message.strip() }}
    {% endif %}
    {{ '\n\n' + message['role'].title() + ': ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '\n\nAssistant:' }}
    {% endif %}
  {% endfor %}
  ````
  """

  # ⟦𓁅𓂇𓍞𓄁⟧ support_list :: auto-generated pointer for public function support_list
  def support_list() do
    [{~r"^falcon.*instruct.*$", 1}]
  end

  defp format_role(role) do
    role |> to_string() |> String.capitalize()
  end

  defp clean_content(content) do
    content
    |> String.trim()
    |> String.replace("\r\n", "\n")
    |> String.replace("\n\n", "\n")
  end

  # ⟦𓁋𓃖𓅮𓃓⟧ extract_response :: auto-generated pointer for public function extract_response
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

  # ⟦𓐠𓄥𓍣𓇂⟧ to_context :: auto-generated pointer for public function to_context
  def to_context(thread, _model, options) do
    {system_message, loop_messages} = case Enum.at(thread, 0) do
      %{role: :system, content: content} -> 
        {String.trim(content), Enum.drop(thread, 1)}
      _ -> 
        {"", thread}
    end

    lines = loop_messages
            |> Enum.with_index()
            |> Enum.map(
                 fn
                   {msg = %{role: role}, index} ->
                     if role in [:user, :assistant] do
                       expected_role = if rem(index, 2) == 0, do: :user, else: :assistant
                       if role != expected_role && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                     end
                     
                     prefix = if index == 0 && system_message != "", do: system_message, else: ""
                     prefix <> "\n\n#{format_role(role)}: #{clean_content(msg.content)}"

                   {msg, index} ->
                     unless options[:strict] == false or options[:expanded_roles] do
                       raise ExLLama.ChatTemplate.Exception, message: "Only user, assistant, and system roles are supported", handler: __MODULE__, entry: msg, row: index
                     end
                     "\n\n#{format_role(msg.role)}: #{clean_content(msg.content)}"
                 end
               ) |> Enum.join("")
    
    result = if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
      lines <> "\n\nAssistant:"
    else
      lines
    end
    
    {:ok, String.trim_leading(result)}
  end
end
