defmodule ExLLama.ChatTemplate.QwenInstruct do
  @moduledoc """
  based on: [https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/qwen2.5-instruct.jinja]
  
  Qwen2.5 Instruct template with extensive tool support.
  Default system message: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  """

  @default_system_message "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

  def support_list() do
    [{~r"^.*qwen.*2\.5.*instruct.*$", 1}, {~r"^.*qwen2\.5.*$", 1}]
  end

  defp format_tools_system_message(nil, first_message) do
    base_msg = case first_message do
      %{role: :system, content: content} -> content
      _ -> @default_system_message
    end
    
    "<|im_start|>system\n#{base_msg}<|im_end|>\n"
  end

  defp format_tools_system_message(tools, first_message) when is_list(tools) do
    base_msg = case first_message do
      %{role: :system, content: content} -> content
      _ -> @default_system_message
    end
    
    tools_json = tools |> Enum.map(fn tool ->
      case Jason.encode(tool) do
        {:ok, json} -> json
        _ -> inspect(tool)
      end
    end) |> Enum.join("\n")
    
    """
    <|im_start|>system
    #{base_msg}

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    #{tools_json}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call><|im_end|>
    """
  end

  defp format_message(%{role: :user, content: content}) do
    "<|im_start|>user\n#{content}<|im_end|>\n"
  end

  defp format_message(%{role: :system, content: content}) do
    "<|im_start|>system\n#{content}<|im_end|>\n"
  end

  defp format_message(%{role: :assistant, content: content, tool_calls: nil}) do
    "<|im_start|>assistant\n#{content}<|im_end|>\n"
  end

  defp format_message(%{role: :assistant, content: content, tool_calls: []}) do
    "<|im_start|>assistant\n#{content}<|im_end|>\n"
  end

  defp format_message(%{role: :assistant} = msg) do
    content_part = if msg[:content], do: "\n#{msg.content}", else: ""
    
    tool_calls_part = (msg[:tool_calls] || [])
                      |> Enum.map(fn tool_call ->
                        # Handle both direct tool_call and wrapped in function key
                        call = if tool_call[:function], do: tool_call.function, else: tool_call
                        
                        args_json = case Jason.encode(call.arguments) do
                          {:ok, json} -> json
                          _ -> inspect(call.arguments)
                        end
                        "\n<tool_call>\n{\"name\": \"#{call.name}\", \"arguments\": #{args_json}}\n</tool_call>"
                      end)
                      |> Enum.join("")
    
    "<|im_start|>assistant#{content_part}#{tool_calls_part}<|im_end|>\n"
  end

  defp format_message(%{role: :tool, content: content}) do
    # Tool messages are wrapped in user messages with tool_response tags
    "\n<tool_response>\n#{content}\n</tool_response>"
  end

  defp format_message(_), do: ""

  def extract_response(responses, model, options) do
    with {:ok, model_name} <- ExLLama.Model.__model_name__(model) do
      choices = responses
                |> Enum.with_index()
                |> Enum.map(
                     fn
                       {{tokens, x}, index} ->
                         content = x 
                                   |> String.trim()
                                   |> String.trim_trailing("<|im_end|>")
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
    tools = options[:tools]
    
    # Handle system message and tools
    {system_handled, remaining_messages} = case {tools, Enum.at(thread, 0)} do
      {nil, %{role: :system}} -> 
        # No tools, but has system message - it will be formatted normally
        {false, thread}
      {nil, _} -> 
        # No tools, no system message - add default
        {true, thread}
      {_, %{role: :system}} -> 
        # Has tools and system message - use it in tools system
        {true, Enum.drop(thread, 1)}
      {_, _} -> 
        # Has tools but no system message - use default
        {true, thread}
    end

    # Format the initial system message (with or without tools)
    system_part = if system_handled do
      format_tools_system_message(tools, Enum.at(thread, 0))
    else
      ""
    end

    # Group consecutive tool messages
    grouped_messages = remaining_messages
                       |> Enum.reduce([], fn msg, acc ->
                         case {msg.role, acc} do
                           {:tool, [{:tool_group, tools} | rest]} ->
                             # Add to existing tool group
                             [{:tool_group, tools ++ [msg]} | rest]
                           {:tool, acc} ->
                             # Start new tool group
                             [{:tool_group, [msg]} | acc]
                           {_, acc} ->
                             # Regular message
                             [msg | acc]
                         end
                       end)
                       |> Enum.reverse()

    # Format all messages
    messages_part = grouped_messages
                    |> Enum.map(fn
                      {:tool_group, tool_msgs} ->
                        tool_responses = tool_msgs
                                         |> Enum.map(&format_message/1)
                                         |> Enum.join("")
                        "<|im_start|>user#{tool_responses}<|im_end|>\n"
                      msg ->
                        if not system_handled or msg.role != :system or remaining_messages != thread do
                          format_message(msg)
                        else
                          ""
                        end
                    end)
                    |> Enum.join("")
    
    result = system_part <> messages_part
    
    if options[:add_generation_prompt] && Enum.at(thread, -1)[:role] != :assistant do
      {:ok, result <> "<|im_start|>assistant\n"}
    else
      {:ok, result}
    end
  end
end