defmodule ExLLama.ChatTemplate.GraniteInstruct do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/granite-3.0-instruct.jinja]
  ```jinja
  {%- if tools %}
      {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}
      {%- for tool in tools %}
          {{- tool | tojson(indent=4) }}
          {%- if not loop.last %}
              {{- '\n\n' }}
          {%- endif %}
      {%- endfor %}
      {{- '<|end_of_text|>\n' }}
  {%- endif %}

  {%- for message in messages %}
      {%- if message['role'] == 'system' %}
          {{- '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
      {%- elif message['role'] == 'user' %}
          {{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
      {%- elif message['role'] == 'assistant' %}
          {{- '<|start_of_role|>assistant<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
      {%- elif message['role'] == 'assistant_tool_call' %}
          {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}
      {%- elif message['role'] == 'tool_response' %}
          {{- '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
      {%- endif %}

      {%- if loop.last and add_generation_prompt %}
          {{- '<|start_of_role|>assistant<|end_of_role|>' }}
      {%- endif %}
  {%- endfor %}
  ```
  """

  def support_list() do
    [{~r"^.*granite.*3.*$", 1}, {~r"^.*granite.*instruct.*$", 1}]
  end

  defp format_message(message) do
    case message.role do
      :system -> "<|start_of_role|>system<|end_of_role|>#{message.content}<|end_of_text|>\n"
      :user -> "<|start_of_role|>user<|end_of_role|>#{message.content}<|end_of_text|>\n"
      :assistant -> "<|start_of_role|>assistant<|end_of_role|>#{message.content}<|end_of_text|>\n"
      :assistant_tool_call -> "<|start_of_role|>assistant<|end_of_role|><|tool_call|>#{message.content}<|end_of_text|>\n"
      :tool_response -> "<|start_of_role|>tool_response<|end_of_role|>#{message.content}<|end_of_text|>\n"
      _ -> ""
    end
  end

  defp format_tools(tools) when is_list(tools) and length(tools) > 0 do
    tool_strings = tools
                   |> Enum.map(fn tool -> 
                     case Jason.encode(tool, pretty: true) do
                       {:ok, json} -> json
                       _ -> inspect(tool)
                     end
                   end)
                   |> Enum.join("\n\n")
    
    "<|start_of_role|>available_tools<|end_of_role|>\n#{tool_strings}<|end_of_text|>\n"
  end
  defp format_tools(_), do: ""

  def extract_response(responses, model, options) do
    with {:ok, model_name} <- ExLLama.Model.__model_name__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         content = x 
                                   |> String.trim()
                                   |> String.trim_trailing("<|end_of_text|>")
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

  def to_context(thread, _model, options) do
    # Handle tools if present
    tools_text = format_tools(options[:tools])
    
    # Format messages
    lines = thread
            |> Enum.map(
                 fn msg ->
                   if msg.role in [:system, :user, :assistant, :assistant_tool_call, :tool_response] do
                     format_message(msg)
                   else
                     unless options[:strict] == false or options[:expanded_roles] do
                       raise ExLLama.ChatTemplate.Exception, 
                         message: "Only system, user, assistant, assistant_tool_call, and tool_response roles are supported", 
                         handler: __MODULE__, 
                         entry: msg, 
                         row: 0
                     end
                     ""
                   end
                 end
               ) |> Enum.join("")
    
    result = tools_text <> lines
    
    if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
      {:ok, result <> "<|start_of_role|>assistant<|end_of_role|>"}
    else
      {:ok, result}
    end
  end
end