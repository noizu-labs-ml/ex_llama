defmodule ExLLama.ChatTemplate.Alpaca do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/alpaca.jinja]
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
        {{ '### Instruction:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### Response:\n' + message['content'].strip() + eos_token + '\n\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '### Instruction:\n' }}
    {% endif %}
  {% endfor %}
  ````
  """

  # ⟦𓈨𓈖𓍉𓍨⟧ support_list :: auto-generated pointer for public function support_list
  def support_list() do
    [{~r"^alpaca.*$", 1}]
  end

  # ⟦𓀋𓋙𓋷𓆪⟧ extract_response :: auto-generated pointer for public function extract_response
  def extract_response(responses, model, options) do
    with {:ok, eos_token} <- ExLLama.Model.__eos__(model),
         {:ok, model_name} <- ExLLama.Model.__model_name__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         content = x 
                                   |> String.trim()
                                   |> String.trim_trailing(eos_token)
                                   |> String.trim_trailing("\n\n")
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

  # ⟦𓍥𓅬𓍧𓌭⟧ to_context :: auto-generated pointer for public function to_context
  def to_context(thread, model, options) do
    with {:ok, bos_token} <- ExLLama.Model.__bos__(model),
         {:ok, eos_token} <- ExLLama.Model.__eos__(model) do
      
      {system_message, loop_messages} = case Enum.at(thread, 0) do
        %{role: :system, content: content} -> 
          {String.trim(content) <> "\n\n", Enum.drop(thread, 1)}
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
                       "### Instruction:\n#{String.trim(msg.content)}\n\n"

                     {msg = %{role: :assistant}, index} ->
                       if rem(index, 2) != 1 && options[:strict] != false do
                         raise ExLLama.ChatTemplate.Exception, message: "Conversation roles must alternate user/assistant/user/assistant/...", handler: __MODULE__, entry: msg, row: index
                       end
                       "### Response:\n#{String.trim(msg.content)}#{eos_token}\n\n"

                     {msg, index} ->
                       unless options[:strict] == false or options[:expanded_roles] do
                         raise ExLLama.ChatTemplate.Exception, message: "Only user, assistant, and system roles are supported", handler: __MODULE__, entry: msg, row: index
                       end
                       ""
                   end
                 ) |> Enum.join("")
      
      result = bos_token <> system_message <> lines
      
      if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] == :user do
        {:ok, result <> "### Response:\n"}
      else
        {:ok, result}
      end
    end
  end
end
