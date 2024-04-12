defmodule ExLLamaTest do
  use ExUnit.Case

  test "Default Session Options" do
      {:ok, sut} = ExLLama.Session.default_options()
      assert sut.__struct__ == ExLLama.SessionOptions
  end

  test "Create Session" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, session} = ExLLama.create_session(llama)
    assert session.__struct__ == ExLLama.Session
  end

  test "Load Model" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    assert llama.__struct__ == ExLLama.Model
  end

  def receive_text(acc \\ []) do
    receive do
      x = {:ok, _} -> Enum.reverse([x|acc])
      x = {:error, _} -> Enum.reverse([x|acc])
      :fin ->
        Enum.reverse(acc)
      x ->
        receive_text([x | acc])
    end
  end

  test "Async complete_with" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, options} = ExLLama.Session.default_options()
    {:ok, session} = ExLLama.create_session(llama, %{options| seed: 2})
    ExLLama.advance_context(session, "<|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n")
    ExLLama.Session.start_completing_with(session, %{max_tokens: 512})
    r = receive_text()
    assert r == [" Good", "bye", "</", "s", ">", ""]
  end

  test "Advance Context" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, options} = ExLLama.Session.default_options()

    {:ok, session} = ExLLama.create_session(llama, %{options| seed: 2})
    ExLLama.advance_context(session, "<|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n")
    {:ok, context} = ExLLama.Session.context(session)
    {:ok, as_str} = ExLLama.Model.decode_tokens(llama, context)
    # There is a bug in advance_context in llama_cpp that injects a space
    assert as_str == " <|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n"
    {:ok, %{content: response}} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
     ExLLama.advance_context(session, response <> "\n<|user|>\n Say Apple.</s>\n<|assistant|>\n")
    {:ok, %{content: response}} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
    ExLLama.advance_context(session, response <> "\n<|user|>\n What did you just say?.</s>\n<|assistant|>\n")
    {:ok, %{content: response}} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
   assert response =~ "Apple"
    {:ok, context} = ExLLama.Session.context(session)
    {:ok, as_str} = ExLLama.Model.decode_tokens(llama, context)
    assert as_str == " <|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n Goodbye</s>\n<|user|>\n Say Apple.</s>\n<|assistant|>\n Apple</s>\n<|user|>\n What did you just say?.</s>\n<|assistant|>\n"
  end

  test  "Chat Completion" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    thread = [
      %{role: :user, content: "Say Hello. And only hello. Example \"Hello\"."},
      %{role: :assistant, content: "Hello"},
      %{role: :user, content: "Repeat what you just said."},
      %{role: :assistant, content: "Hello"},
      %{role: :user, content: "Say Goodbye."},
      %{role: :assistant, content: "Goodbye"},
      %{role: :user, content: "Say Apple."},
      %{role: :assistant, content: "Apple"},
      %{role: :user, content: "What did you just say?."},
    ]

    # After stripping </s> completion_tokens are actually 3, although it's useful to know how many tokens were generated.
    {:ok, response} = ExLLama.chat_completion(llama, thread, [seed: 2, choices: 2])
    assert response == %ExLLama.ChatCompletion{
             choices: [
               %ExLLama.ChatCompletion.Choice{finish_reason: :stop, index: 0, message: "Apple"},
               %ExLLama.ChatCompletion.Choice{finish_reason: :stop, index: 1, message: "Apple"}
             ],
             id: nil,
             model: "./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
             seed: 2,
             usage: %ExLLama.ChatCompletion.Usage{prompt_tokens: 143, total_tokens: 147, completion_tokens: 4},
             vsn: 1.0
           }
  end


end
