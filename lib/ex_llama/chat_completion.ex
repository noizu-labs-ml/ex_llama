defmodule ExLLama.ChatCompletion.Choice do
  defstruct [
    index: nil,
    message: nil,
    finish_reason: nil,
  ]
end

defmodule ExLLama.ChatCompletion.Usage do
  defstruct [
    prompt_tokens: nil,
    total_tokens: nil,
    completion_tokens: nil,
  ]
end



defmodule ExLLama.ChatCompletion do
  @vsn 1.0
  defstruct [
    id: nil,
    model: nil,
    seed: nil,
    choices: nil,
    usage: nil,
    vsn: @vsn
  ]

  @type t :: %__MODULE__{
               id: String.t,
               model: String.t,
               seed: String.t,
               choices: [ExLLama.ChatCompletion.Choice.t],
               usage: ExLLama.ChatCompletion.Usage.t,
               vsn: float
             }
end
