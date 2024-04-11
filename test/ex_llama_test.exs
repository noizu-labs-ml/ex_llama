defmodule ExLLamaTest do
  use ExUnit.Case

  test "Default Session Options" do
      sut = ExLLama.default_session_options()
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
    {:ok, session} = ExLLama.create_session(llama)
    advance = ExLLama.advance_context(session, "<|user|>\nSay Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>Hello</s>\n<|user|>\nRepeat what you just said.</s>\n<|assistant|>Hello</s>\n<|user|>\nSay Goodbye.</s>\n<|assistant|>")
    IO.inspect(advance, label: "Advance: #{inspect "<|user|>\nSay Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>Hello</s>\n<|user|>\nRepeat what you just said.</s>\n<|assistant|>Hello</s>\n<|user|>\nSay Goodbye.</s>\n<|assistant|>"} -->")
    {:ok, response} = ExLLama.complete(session)
    IO.inspect(response, label: "Completion 1")
    advance = ExLLama.advance_context(session, "\n" <> response <> "\n<|user|>\nSay Apple.</s>\n<|assistant|>")
    IO.inspect(advance, label: "Advance: #{inspect  response <> "\n<|user|>\nSay Apple.</s>\n<|assistant|>"} -->")
    {:ok, response} = ExLLama.complete(session)
    IO.inspect(response, label: "Completion 2")
    advance = ExLLama.advance_context(session, "\n" <> response <> "\n<|user|>\nWhat did you just say?.</s>\n<|assistant|>")
    IO.inspect(advance, label: "Advance #{inspect  response <> "\n<|user|>\nWhat did you just say?.</s>\n<|assistant|>"} -->")
    {:ok, response} = ExLLama.complete(session)
    IO.inspect(response, label: "Completion 3")
    IO.inspect(llama)
  end

end
