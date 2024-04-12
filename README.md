ExLLama: LlammaCpp.rs NIF wrapper for Elixir/Erlang.
=======

This is an Alpha Library for loading and interacting with models via the llama_cpp rust client exposed as nif extensions. 
Inspired By [llama_cpp_ex](https://github.com/jeregrine/llama_cpp_ex)


## Getting Started
1. Add the `ex_llama` dependency to your `mix.exs` file:

```elixir
def deps do 
  [
    {:ex_llama, "~> 0.0.1"}
  ]

end
```


## Chat Completion 
As of this build only `<|role|>messsage</s>` format chat completion is supported, such as used by tiny llama. 


```elixir 

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

    {:ok, response} = ExLLama.chat_completion(llama, thread, %{seed: 2})
    # response = %{
    #         choices: [
    #           %{reason: :end, role: "assistant", content: "Apple"},
    #           %{reason: :end, role: "assistant", content: "Apple"},
    #           %{reason: :end, role: "assistant", content: "Apple"}
    #         ]
    # }

```


## Simple Completion (direct)
```elixir
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, options} = ExLLama.Session.default_options()
    {:ok, session} = ExLLama.create_session(llama, %{options| seed: 2})
    ExLLama.advance_context(session, "<|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n")
    {:ok, response} = ExLLama.completion(session, 512, "</s>\n*")
    response = String.trim_leading(response)
    # "Goodbye.</s>"
```

## Streaming Completion (final mechanism will be replaced with a Stream
```elixir

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

#...
    {:ok, llama} = ExLLama.load_model("./test/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    {:ok, options} = ExLLama.Session.default_options()
    {:ok, session} = ExLLama.create_session(llama, %{options| seed: 2})
    ExLLama.advance_context(session, "<|user|>\n Say Hello. And only hello. Example \"Hello\".</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Repeat what you just said.</s>\n<|assistant|>\n Hello</s>\n<|user|>\n Say Goodbye.</s>\n<|assistant|>\n")
    ExLLama.Session.start_completing_with(session, %{max_tokens: 512})
    receive_text()


```

