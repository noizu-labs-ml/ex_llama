defmodule ExLLama.ClassifierContractTest do
  use ExUnit.Case, async: true
  alias GenAI.Classifier.Question
  alias ExLLama.Classifier.Decision

  test "choice order and stable IDs are retained; first maximum wins" do
    question = %Question{id: "q", choices: [%{id: "z", label: "Same"}, %{id: "a", label: "Same"}]}
    assert {:ok, answer} = Decision.answer(question, [0.5, 0.5])
    assert answer.choice_id == "z"
    assert answer.index == 0
    assert answer.selected_probability == 0.5
    assert answer.provider_confidence == 0.5
    assert answer.confidence_metadata.meaning == :maximum_option_probability
  end

  test "boolean distribution uses false then true and threshold argmax" do
    question = %Question{id: "b", type: :boolean}

    assert {:ok, %{value: true, choice_id: "true", index: 1}} =
             Decision.answer(question, [0.1, 0.9])

    assert {:ok, %{value: false, choice_id: "false", index: 0}} =
             Decision.answer(question, [0.5, 0.5])
  end

  test "ordinal grade and expected score remain separate on arbitrary scales" do
    question = %Question{
      id: "s",
      type: :score,
      rubric: [
        %{id: "poor", label: "Poor", score: 0},
        %{id: "fair", label: "Fair", score: 2},
        %{id: "good", label: "Good", score: 10}
      ]
    }

    assert {:ok, answer} = Decision.answer(question, [0.1, 0.2, 0.7])
    assert answer.value == 10
    assert answer.grade == "good"
    assert abs(answer.expected_score - 7.4) < 1.0e-9
  end

  test "bad shapes, probability values, sums and duplicate criteria are rejected" do
    question = %Question{id: "q", choices: ["a", "b"]}

    for distribution <- [nil, [1], [0.2, 0.2], [-1, 2], ["x", 1]] do
      assert {:error, _} = Decision.answer(question, distribution)
    end

    assert {:error, _} = Decision.answer(%{question | choices: ["a", "a"]}, [0.5, 0.5])
  end

  test "256 options accepted and 257 rejected before native inference" do
    question = %Question{id: "q", choices: Enum.map(1..256, &Integer.to_string/1)}
    assert {:ok, %{index: 0}} = Decision.answer(question, [1.0 | List.duplicate(0.0, 255)])
    question = %{question | choices: question.choices ++ ["257"]}
    assert {:error, _} = Decision.answer(question, [1.0 | List.duplicate(0.0, 256)])
  end

  test "text prompt matches upstream format and disabled-thinking generation suffix" do
    prompt = ExLLama.Classifier.prompt(" State ", "Question?", ["Yes", "No"])

    assert prompt ==
             "<bos><|turn>user\nState \n\n---\n\nQUESTION: Question?\n\nOPTIONS:\n1. Yes\n2. No\n\n" <>
               "Reply with only the number of the correct option (1-2).\nOutput a single number and nothing else." <>
               "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

    assert length(String.split(prompt, "<bos>")) == 2
    refute prompt =~ "<|think|>"
  end

  test "adapter declares text and sequential questions explicitly" do
    capabilities = ExLLama.Classifier.classifier_capabilities()
    assert capabilities.modalities == [:text]
    assert capabilities.native_multi_question == false
    assert capabilities.max_choices == 256
  end
end
