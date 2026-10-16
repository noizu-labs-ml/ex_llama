defmodule ExLLama.Classifier.Decision do
  @moduledoc false
  alias GenAI.Classifier.{Question, Result, Validation}
  alias GenAI.Classifier.Result.Answer

  # Pure conversion, shared by native inference and contract tests. The first
  # maximum wins, matching numpy.argmax in the upstream reference implementation.
  @doc false
  @spec answer(Question.t() | map(), [number()]) :: {:ok, Answer.t()} | {:error, term()}
  def answer(question, distribution) do
    with {:ok, question} <- Question.normalize(question),
         true <- is_list(distribution),
         entries = Question.entries(question),
         true <- length(distribution) == length(entries) and length(entries) in 2..256,
         true <- Enum.all?(distribution, &Validation.probability?/1),
         true <- abs(Enum.sum(distribution) - 1) <= 1.0e-6 do
      {probability, index} =
        Enum.reduce(Enum.with_index(distribution), {hd(distribution), 0}, fn
          {probability, index}, {best, _} when probability > best -> {probability, index}
          _, best -> best
        end)

      entry = Enum.at(entries, index)

      answer = %Answer{
        question_id: question.id,
        type: question.type,
        index: index,
        choice_id: entry.id,
        distribution: distribution,
        selected_probability: probability,
        provider_confidence: probability,
        confidence_metadata: %{meaning: :maximum_option_probability},
        metadata: %{pooling: :last_token, normalization: :learned_mu_sd, head_precision: :fp32}
      }

      answer =
        case question.type do
          :boolean ->
            %{answer | value: index == 1}

          :choice ->
            %{answer | value: entry.id}

          :score ->
            expected =
              Enum.zip(entries, distribution)
              |> Enum.reduce(0.0, fn {entry, probability}, total ->
                total + entry.score * probability
              end)

            %{answer | value: entry.score, grade: entry.id, expected_score: expected}
        end

      request = %GenAI.Classifier.Request{input: "", questions: [question]}

      with {:ok, %Result{answers: [answer]}} <- Result.normalize(%{answers: [answer]}, request),
           do: {:ok, answer}
    else
      false -> {:error, :invalid_classifier_distribution}
      {:error, _} = error -> error
    end
  end
end
