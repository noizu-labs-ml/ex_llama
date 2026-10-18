defmodule ExLLama.Classifier do
  @moduledoc """
  Text-only native Jev-Omni inference with the independent trained decision head.
  Each typed question performs one backbone forward pass. Generation and ordinary
  normalized embeddings are separate operations. Load once and pass the immutable
  classifier in `classifier:`; every question owns a private native context.
  """
  @behaviour GenAI.Classifier.ProviderBehaviour
  alias GenAI.Classifier.{Question, Request, Result}

  @enforce_keys [:model, :head, :hidden]
  defstruct [:model, :head, :hidden, :model_id, context_options: %{}, metadata: %{}]

  @type t :: %__MODULE__{
          model: struct(),
          head: reference(),
          hidden: pos_integer(),
          model_id: String.t() | nil,
          context_options: map(),
          metadata: map()
        }

  @impl true
  def classifier_capabilities do
    %{
      types: [:choice, :boolean, :score],
      modalities: [:text],
      max_choices: 256,
      native_multi_question: false
    }
  end

  @doc "Load a text GGUF backbone and its portable trained decision head."
  @spec load(String.t(), String.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load(model_path, head_path, options \\ []) do
    model_options = ExLLama.ModelOptions.new(Keyword.get(options, :model_options, []))

    with {:ok, head} <- ExLLama.Nif.classifier_load_head(head_path),
         {:ok, %{hidden: hidden}} <- ExLLama.Nif.classifier_head_info(head),
         {:ok, model} <- ExLLama.load_model(model_path, model_options),
         {:ok, %{n_embd_out: ^hidden}} <- ExLLama.Nif.model_info(model.resource) do
      {:ok,
       %__MODULE__{
         model: model,
         head: head,
         hidden: hidden,
         model_id: Keyword.get(options, :model_id, model_path),
         context_options: Map.new(Keyword.get(options, :context_options, [])),
         metadata: %{
           backbone: model_path,
           head_format: "JEVH0001",
           head_sha256: file_hash(head_path)
         }
       }}
    else
      {:ok, %{n_embd_out: _}} -> {:error, :classifier_hidden_dimension_mismatch}
      {:error, _} = error -> error
    end
  end

  @doc "Return the Core model identity of a loaded classifier."
  @spec model(t()) :: struct()
  def model(%__MODULE__{} = classifier),
    do: %GenAI.Model{provider: __MODULE__, model: classifier.model_id || classifier.model.name}

  @impl true
  def classify(request, options) do
    with {:ok, request} <- Request.normalize(request),
         true <- request.modality == :text,
         %__MODULE__{} = classifier <- options[:classifier],
         :ok <- matching_model(request.model, classifier),
         {:ok, state} <- encode_state(request.input) do
      started = System.monotonic_time(:microsecond)

      answers =
        Enum.reduce_while(request.questions, {:ok, []}, fn question, {:ok, answers} ->
          case infer_question(classifier, state, question, options) do
            {:ok, answer} -> {:cont, {:ok, [answer | answers]}}
            {:error, _} = error -> {:halt, error}
          end
        end)

      with {:ok, answers} <- answers do
        result = %Result{
          answers: Enum.reverse(answers),
          model: request.model || model(classifier),
          provider: __MODULE__,
          duration: System.monotonic_time(:microsecond) - started,
          metadata:
            Map.merge(classifier.metadata, %{
              duration_unit: :microsecond,
              native_multi_question: false,
              forward_passes: length(request.questions)
            })
        }

        Result.normalize(result, request)
      end
    else
      false -> {:error, {:unsupported_classifier_modality, request.modality}}
      nil -> {:error, :loaded_classifier_required}
      {:error, _} = error -> error
      _ -> {:error, :invalid_loaded_classifier}
    end
  end

  @doc "Reference Jev prompt, including Gemma4's empty thought-channel generation prefix."
  @spec prompt(String.t(), String.t(), [String.t()]) :: String.t()
  def prompt(state, question, labels) do
    choices =
      labels |> Enum.with_index(1) |> Enum.map_join("\n", fn {label, i} -> "#{i}. #{label}" end)

    text =
      "#{state}\n\n---\n\nQUESTION: #{question}\n\nOPTIONS:\n#{choices}\n\n" <>
        "Reply with only the number of the correct option (1-#{length(labels)}).\n" <>
        "Output a single number and nothing else."

    "<bos><|turn>user\n" <>
      String.trim(text) <>
      "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
  end

  defp infer_question(classifier, state, question, options) do
    entries = Question.entries(question)
    labels = if question.type == :boolean, do: ["No", "Yes"], else: Enum.map(entries, & &1.label)
    instructions = question.instructions || default_question(question.type)

    settings =
      Map.merge(classifier.context_options, Map.new(Keyword.get(options, :context_options, [])))

    with true <- length(entries) in 2..256,
         {:ok, features} <-
           ExLLama.Nif.classifier_last_hidden(
             classifier.model.resource,
             prompt(state, instructions, labels),
             settings
           ),
         {:ok, distribution} <-
           ExLLama.Nif.classifier_head_probabilities(classifier.head, features, length(entries)) do
      ExLLama.Classifier.Decision.answer(question, distribution)
    else
      false -> {:error, {:classifier_limit, :max_choices}}
      {:error, _} = error -> error
    end
  end

  defp default_question(:boolean), do: "Is the statement true?"
  defp default_question(:score), do: "Which rubric level best describes the state?"
  defp default_question(:choice), do: "Which option best describes the state?"
  defp encode_state(state) when is_binary(state), do: {:ok, state}
  defp encode_state(state), do: Jason.encode(state)
  defp matching_model(nil, _classifier), do: :ok

  defp matching_model(%GenAI.Model{model: id}, classifier) do
    if id == model(classifier).model, do: :ok, else: {:error, :loaded_classifier_model_mismatch}
  end

  defp file_hash(path),
    do: path |> File.read!() |> then(&:crypto.hash(:sha256, &1)) |> Base.encode16(case: :lower)
end
