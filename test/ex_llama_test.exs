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

  test "Advance Context" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, options} = ExLLama.Session.default_options()

    {:ok, session} = ExLLama.create_session(llama, %{options| seed: 2})
    ExLLama.advance_context(session, "<|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n")
    {:ok, context} = ExLLama.Session.context(session)
    {:ok, as_str} = ExLLama.Model.decode_tokens(llama, context)
    # There is a bug in advance_context in llama_cpp that injects a space
    assert as_str == " <|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n"
    {:ok, response} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
     ExLLama.advance_context(session, response <> "\n<|user|>\n Say Apple.</s>\n<|assistant|>\n")
    {:ok, response} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
    ExLLama.advance_context(session, response <> "\n<|user|>\n What did you just say?.</s>\n<|assistant|>\n")
    {:ok, response} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
   assert response =~ "Apple"
    {:ok, context} = ExLLama.Session.context(session)
    {:ok, as_str} = ExLLama.Model.decode_tokens(llama, context)
    assert as_str == " <|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n Goodbye</s>\n<|user|>\n Say Apple.</s>\n<|assistant|>\n Apple</s>\n<|user|>\n What did you just say?.</s>\n<|assistant|>\n"
  end


end
