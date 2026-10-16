defmodule ExLLama.ClassifierNativeTest do
  use ExUnit.Case, async: false

  alias ExLLama.Nif

  setup_all do
    model_path =
      Path.join([
        to_string(:code.priv_dir(:ex_llama)),
        "models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
      ])

    {:ok, model} = ExLLama.load_model(model_path)
    {:ok, info} = Nif.model_info(model.resource)
    %{model: model, model_path: model_path, hidden: info.n_embd_out}
  end

  test "native head validates files and computes masked probabilities" do
    path = head_file(2, [0.0, 1.0986123])
    assert {:ok, head} = Nif.classifier_load_head(path)
    assert {:ok, %{hidden: 2, classes: 256}} = Nif.classifier_head_info(head)
    assert {:ok, [first, second]} = Nif.classifier_head_probabilities(head, [2.0, -1.0], 2)
    assert_in_delta first, 0.25, 1.0e-6
    assert_in_delta second, 0.75, 1.0e-6
    assert {:error, _} = Nif.classifier_head_probabilities(head, [1.0], 2)
    assert {:error, _} = Nif.classifier_head_probabilities(head, [1.0, 2.0], 257)
    assert {:error, _} = Nif.classifier_load_head(path <> ".missing")
    File.write!(path, "JEVH0001")
    assert {:error, _} = Nif.classifier_load_head(path)
    # The loaded resource owns its parameters independently of the file.
    assert {:ok, [_, _]} = Nif.classifier_head_probabilities(head, [1, 2], 2)
  end

  test "last-token features agree across prompt chunks and independent contexts", context do
    prompt = "<s>" <> String.duplicate("The quick brown fox jumps over the lazy dog. ", 3)
    small = %{n_ctx: 256, n_batch: 16, n_ubatch: 16}
    large = %{n_ctx: 256, n_batch: 128, n_ubatch: 128}

    assert {:ok, chunked} = Nif.classifier_last_hidden(context.model.resource, prompt, small)
    assert {:ok, whole} = Nif.classifier_last_hidden(context.model.resource, prompt, large)
    assert length(chunked) == context.hidden

    Enum.zip(chunked, whole)
    |> Enum.each(fn {left, right} -> assert_in_delta left, right, 1.0e-3 end)

    assert {:ok, other} =
             Nif.classifier_last_hidden(
               context.model.resource,
               prompt <> "Different ending",
               small
             )

    refute other == chunked

    outputs =
      1..2
      |> Task.async_stream(
        fn _ -> Nif.classifier_last_hidden(context.model.resource, prompt, small) end,
        timeout: 30_000
      )
      |> Enum.to_list()

    assert [{:ok, {:ok, ^chunked}}, {:ok, {:ok, ^chunked}}] = outputs
  end

  test "invalid context limits return errors before native evaluation", context do
    for options <- [
          %{n_ctx: 0},
          %{n_ctx: "256"},
          %{n_ctx: 262_145},
          %{n_ctx: 16, n_batch: 32},
          %{n_ctx: 32, n_batch: 16, n_ubatch: 32}
        ] do
      assert {:error, _} = Nif.classifier_last_hidden(context.model.resource, "<s>Hello", options)
    end
  end

  test "loader rejects a head whose width differs from the backbone output", context do
    path = head_file(2, [0.0, 0.0])

    assert {:error, :classifier_hidden_dimension_mismatch} =
             ExLLama.load_classifier(context.model_path, path)
  end

  test "classifier forward passes preserve ordinary generation and embedding contexts", context do
    resource = context.model.resource
    assert {:ok, session} = Nif.create_context(resource, %{n_ctx: 128, n_batch: 64})
    prompt = "<|user|>\nSay Hello.\n<|assistant|>\n"
    options = %{max_tokens: 8, seed: 2, temp: 0.0}
    assert {:ok, before} = Nif.completion(session, prompt, options)

    assert {:ok, _} =
             Nif.classifier_last_hidden(resource, "<s>Classifier input", %{
               n_ctx: 128,
               n_batch: 64,
               n_ubatch: 64
             })

    assert {:ok, after_classifier} = Nif.completion(session, prompt, options)
    assert before == after_classifier

    assert {:ok, embedding_context} =
             Nif.create_context(resource, %{n_ctx: 128, n_batch: 64, embeddings: true})

    assert {:ok, [embedding]} = Nif.embeddings(embedding_context, "Hello world")
    assert length(embedding) == context.hidden
    assert_in_delta Enum.reduce(embedding, 0.0, &(&1 * &1 + &2)), 1.0, 1.0e-5
  end

  defp head_file(hidden, active_biases) do
    path = Path.join(System.tmp_dir!(), "jev-native-#{System.unique_integer([:positive])}.bin")
    on_exit(fn -> File.rm(path) end)
    zeros = :binary.copy(<<0.0::little-float-size(32)>>, hidden)
    ones = :binary.copy(<<1.0::little-float-size(32)>>, hidden)
    weights = :binary.copy(zeros, 256)
    biases = active_biases ++ List.duplicate(10_000.0, 256 - length(active_biases))
    bias = Enum.map(biases, &<<&1::little-float-size(32)>>)

    File.write!(path, [
      "JEVH0001",
      <<hidden::little-32, 256::little-32>>,
      zeros,
      ones,
      weights,
      bias
    ])

    path
  end
end
