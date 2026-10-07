defmodule ExLLama.ChatTemplate.Phi3 do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/phi-3.jinja]
  ```jinja
  {% if messages[0]['role'] == 'system' %}
      {% set offset = 1 %}
  {% else %}
      {% set offset = 0 %}
  {% endif %}

  {% for message in messages %}
      {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
          {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
      {% endif %}

      {{ '<|' + message['role'] + '|>\n' + message['content'] | trim + '<|end|>' + '\n' }}
  {% endfor %}

  {% if add_generation_prompt %}
      {{ '<|assistant|>\n' }}
  {% endif %}
  ```
  """

  # ⟦𓍧𓅷𓉗𓉬⟧ support_list :: auto-generated pointer for public function support_list
  def support_list() do
    [{~r"^.*phi.*3.*$", 1}, {~r"^.*phi-3[^-].*$", 1}]
  end

  defp format_message(message) do
    "<|#{message.role}|>\n#{String.trim(message.content)}<|end|>\n"
  end

  # ⟦𓁀𓏊𓎝𓁈⟧ extract_response :: auto-generated pointer for public function extract_response
  def extract_response(responses, model, options) do
    with {:ok, model_name} <- ExLLama.Model.__model_name__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         content = x 
                                   |> String.trim()
                                   |> String.trim_trailing("<|end|>")
                                   |> String.trim()
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

  # ⟦𓀅𓉄𓊁𓐔⟧ to_context :: auto-generated pointer for public function to_context
  def to_context(thread, _model, options) do
    # Note: Unlike Phi3Small, this template doesn't include the BOS token
    system_offset = if Enum.at(thread, 0)[:role] == :system, do: 1, else: 0
    
    lines = thread
            |> Enum.with_index()
            |> Enum.map(
                 fn
                   {msg = %{role: :system}, 0} ->
                     format_message(msg)
                     
                   {msg = %{role: :system}, index} ->
                     unless options[:strict] == false do
                       raise ExLLama.ChatTemplate.Exception, message: "Only the first message may be from system", handler: __MODULE__, entry: msg, row: index
                     end
                     format_message(msg)

                   {msg = %{role: role}, index} when role in [:user, :assistant] ->
                     expected_user = rem(index - system_offset, 2) == 0
                     is_user = role == :user
                     
                     if is_user != expected_user && options[:strict] != false do
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
      {:ok, lines <> "<|assistant|>\n"}
    else
      {:ok, lines}
    end
  end
end