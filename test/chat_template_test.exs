defmodule ExLLama.ChatTemplateTest do
  use ExUnit.Case

  # Mock model for testing
  def mock_model() do
    %ExLLama.Model{
      resource: nil,
      eos: '</s>',
      bos: '<s>',
      name: "test-model"
    }
  end

  describe "Zephyr template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<|user|>\n Hello</s>\n\n<|assistant|>\n Hi there</s>\n\n<|user|>\n How are you?</s>\n<|assistant|>\n"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<|system|>\n You are a helpful assistant</s>\n\n<|user|>\n Hello</s>\n<|assistant|>\n"
      assert result == expected
    end

    test "raises on invalid role alternation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi"},
        %{role: :user, content: "How are you?"},
        %{role: :user, content: "Hello again"}
      ]
      
      assert_raise ExLLama.ChatTemplate.Exception, fn ->
        ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{strict: true})
      end
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world</s>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.Zephyr.extract_response(responses, mock_model(), options)
      
      assert completion.model == "test-model"
      assert completion.seed == 42
      assert length(completion.choices) == 1
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 5
      assert completion.usage.prompt_tokens == 10
    end
  end

  describe "ChatML template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.ChatML.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.ChatML.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s><|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
      assert result == expected
    end

    test "extracts response correctly" do
      responses = [{8, "Hello world<|im_end|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.ChatML.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
    end
  end

  describe "Alpaca template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Alpaca.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>### Instruction:\nHello\n\n### Response:\nHi there</s>\n\n### Instruction:\nHow are you?\n\n### Response:\n"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Alpaca.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>You are a helpful assistant\n\n### Instruction:\nHello\n\n### Response:\n"
      assert result == expected
    end

    test "does not add generation prompt when last message is assistant" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Alpaca.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>### Instruction:\nHello\n\n### Response:\nHi there</s>\n\n"
      assert result == expected
    end
  end

  describe "AmberChat template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.AmberChat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>###Human: Hello\n###Assistant: Hi there\n###Human: How are you?\n###Assistant:"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.AmberChat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>You are a helpful assistant\n###Human: Hello\n###Assistant:"
      assert result == expected
    end

    test "validates role alternation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :user, content: "Hello again"}
      ]
      
      assert_raise ExLLama.ChatTemplate.Exception, fn ->
        ExLLama.ChatTemplate.AmberChat.to_context(thread, mock_model(), %{})
      end
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.AmberChat.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 5
    end
  end

  describe "FalconInstruct template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.FalconInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "User: Hello\n\nAssistant: Hi there\n\nUser: How are you?\n\nAssistant:"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.FalconInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "You are a helpful assistant\n\nUser: Hello\n\nAssistant:"
      assert result == expected
    end

    test "cleans content with multiple newlines" do
      thread = [
        %{role: :user, content: "Hello\n\nWorld"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.FalconInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "User: Hello\nWorld\n\nAssistant:"
      assert result == expected
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.FalconInstruct.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 5
    end
  end

  describe "ChatQA template" do
    test "formats basic user/assistant conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"},
        %{role: :user, content: "How are you?"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.ChatQA.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>\n\nUser: Hello\n\nAssistant: Hi there\n\nUser: How are you?\n\nAssistant:"
      assert result == expected
    end

    test "handles system and context messages" do
      thread = [
        %{role: :system, content: "You are a helpful assistant"},
        %{role: :context, content: "Previous context here"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.ChatQA.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s>System: You are a helpful assistant\n\nPrevious context here\n\nUser: Hello\n\nAssistant:"
      assert result == expected
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.ChatQA.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 5
    end
  end

  describe "GraniteInstruct template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.GraniteInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
      assert result == expected
    end

    test "handles tool-related roles" do
      thread = [
        %{role: :user, content: "What's the weather?"},
        %{role: :assistant_tool_call, content: "{\"name\": \"get_weather\", \"args\": {}}"},
        %{role: :tool_response, content: "Sunny, 22°C"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.GraniteInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "<|start_of_role|>assistant<|end_of_role|><|tool_call|>"
      assert result =~ "<|start_of_role|>tool_response<|end_of_role|>"
    end

    test "extracts response correctly" do
      responses = [{8, "Hello world<|end_of_text|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.GraniteInstruct.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 8
    end
  end

  describe "Llama3Instruct template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Llama3Instruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      assert result == expected
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Llama3Instruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "<|start_header_id|>system<|end_header_id|>"
      assert result =~ "<|start_header_id|>user<|end_header_id|>"
    end

    test "extracts response correctly" do
      responses = [{7, "Hello world<|eot_id|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.Llama3Instruct.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 7
    end
  end

  describe "Phi3Small template" do
    test "formats with BOS token" do
      thread = [
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Phi3Small.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<s><|user|>\nHello<|end|>\n<|assistant|>\n"
      assert result == expected
    end

    test "extracts response correctly" do
      responses = [{6, "Hello world<|end|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.Phi3Small.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 6
    end
  end

  describe "Phi3 template" do
    test "formats without BOS token" do
      thread = [
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Phi3.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      expected = "<|user|>\nHello<|end|>\n<|assistant|>\n"
      assert result == expected
    end

    test "extracts response correctly" do
      responses = [{6, "Hello world<|end|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.Phi3.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 6
    end
  end

  describe "QwenInstruct template" do
    test "formats basic conversation with default system message" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.QwenInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "You are Qwen, created by Alibaba Cloud"
      assert result =~ "<|im_start|>user\nHello<|im_end|>"
      assert result =~ "<|im_start|>assistant\nHi there<|im_end|>"
    end

    test "handles custom system message" do
      thread = [
        %{role: :system, content: "Custom system prompt"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.QwenInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "Custom system prompt"
      refute result =~ "You are Qwen"
    end

    test "formats tool messages" do
      thread = [
        %{role: :user, content: "What's 2+2?"},
        %{role: :assistant, content: "Let me calculate", tool_calls: [%{name: "calculator", arguments: %{expr: "2+2"}}]},
        %{role: :tool, content: "4"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.QwenInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "<tool_call>"
      assert result =~ "\"name\": \"calculator\""
      assert result =~ "<tool_response>"
    end
  end

  describe "GemmaInstruct template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.GemmaInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "Hello"
      assert result =~ "Hi there"
      assert result =~ "<start_of_turn>"
      assert result =~ "<end_of_turn>"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.GemmaInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "You are helpful"
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.GemmaInstruct.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
      assert completion.usage.completion_tokens == 5
    end
  end

  describe "Llama2Chat template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.LLama2Chat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "[INST]"
      assert result =~ "[/INST]"
      assert result =~ "Hello"
      assert result =~ "Hi there"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.LLama2Chat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "<<SYS>>"
      assert result =~ "<</SYS>>"
      assert result =~ "You are helpful"
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world</s>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.LLama2Chat.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
    end
  end

  describe "MistralInstruct template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.MistralInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "[INST]"
      assert result =~ "[/INST]"
      assert result =~ "Hello"
      assert result =~ "Hi there"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.MistralInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "You are helpful"
      assert result =~ "[INST]"
    end

    test "does not add generation prompt when last is assistant" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.MistralInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      refute result =~ ~r/\[\/INST\]$/
    end
  end

  describe "OpenChat template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.OpenChat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "GPT4 Correct User:"
      assert result =~ "GPT4 Correct Assistant:"
      assert result =~ "Hello"
      assert result =~ "Hi there"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.OpenChat.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "You are helpful"
    end

    test "extracts response correctly" do
      responses = [{8, "Hello world<|end_of_turn|>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.OpenChat.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
    end
  end

  describe "Saiga template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Saiga.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "user"
      assert result =~ "Hello"
      assert result =~ "Hi there"
      assert result =~ "bot"  # Saiga uses 'bot' in generation prompt
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Saiga.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "system"
      assert result =~ "You are helpful"
    end

    test "enforces role alternation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :user, content: "Hello again"}
      ]
      
      assert_raise ExLLama.ChatTemplate.Exception, fn ->
        ExLLama.ChatTemplate.Saiga.to_context(thread, mock_model(), %{})
      end
    end
  end

  describe "SolarInstruct template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.SolarInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "### User:"
      assert result =~ "### Assistant:"
      assert result =~ "Hello"
      assert result =~ "Hi there"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.SolarInstruct.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "### System:"
      assert result =~ "You are helpful"
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.SolarInstruct.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
    end
  end

  describe "Vicuna template" do
    test "formats basic conversation" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :assistant, content: "Hi there"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Vicuna.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "USER:"
      assert result =~ "ASSISTANT:"
      assert result =~ "Hello"
      assert result =~ "Hi there"
    end

    test "handles system message" do
      thread = [
        %{role: :system, content: "You are helpful"},
        %{role: :user, content: "Hello"}
      ]
      
      {:ok, result} = ExLLama.ChatTemplate.Vicuna.to_context(thread, mock_model(), %{add_generation_prompt: true})
      
      assert result =~ "You are helpful"
    end

    test "extracts response correctly" do
      responses = [{5, "Hello world</s>"}]
      options = %{max_tokens: 100, prompt_tokens: 10, seed: 42}
      
      {:ok, completion} = ExLLama.ChatTemplate.Vicuna.extract_response(responses, mock_model(), options)
      
      assert Enum.at(completion.choices, 0).message.content == "Hello world"
    end
  end

  describe "Template selection" do
    test "picks handler based on template option" do
      handler = ExLLama.ChatTemplate.pick_handler(mock_model(), [template: ExLLama.ChatTemplate.ChatML])
      assert handler == ExLLama.ChatTemplate.ChatML
    end

    test "defaults to Zephyr when no template specified" do
      handler = ExLLama.ChatTemplate.pick_handler(mock_model(), [])
      assert handler == ExLLama.ChatTemplate.Zephyr
    end
  end

  describe "Error handling" do
    test "provides meaningful error messages" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :unknown, content: "Test"}
      ]
      
      try do
        ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{})
      rescue
        e in ExLLama.ChatTemplate.Exception ->
          assert e.handler == ExLLama.ChatTemplate.Zephyr
          assert e.row == 1
          assert e.entry == %{role: :unknown, content: "Test"}
      end
    end
  end

  describe "Strict mode" do
    test "allows non-standard roles with strict: false" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :tool, content: "Function result"},
        %{role: :assistant, content: "I see"}
      ]
      
      {:ok, _result} = ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{strict: false})
    end

    test "allows invalid alternation with strict: false" do
      thread = [
        %{role: :user, content: "Hello"},
        %{role: :user, content: "Hello again"}
      ]
      
      {:ok, _result} = ExLLama.ChatTemplate.Zephyr.to_context(thread, mock_model(), %{strict: false})
    end
  end
end