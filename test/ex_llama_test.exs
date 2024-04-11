defmodule ExLLamaTest do
  use ExUnit.Case

  test "Load Model" do
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    assert llama.__struct__ == ExLLama.Model
  end

end
