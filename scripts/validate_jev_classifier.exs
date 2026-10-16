# Run from the feature checkout after compiling the real NIF. Model paths are
# positional arguments; this script makes no downloads and uses no credentials.
[model_path, head_path, output_path | _] = System.argv()
gpu_layers = System.get_env("JEV_GPU_LAYERS", "99") |> String.to_integer()

{:ok, classifier} =
  ExLLama.load_classifier(model_path, head_path,
    model_id: "akhilaaa3/Jev-Omni",
    model_options: [n_gpu_layers: gpu_layers],
    context_options: [n_ctx: 2048, n_batch: 256, n_ubatch: 256]
  )

cases = [
  {"meeting", "The meeting starts at 10 AM. It is now 9 AM.", "Has the meeting started?",
   ["Yes", "No"]},
  {"refund", "Please refund my subscription payment.", "What is the customer asking for?",
   ["A refund", "Technical help", "A new account"]},
  {"neutral", "The sky is blue.", "Does this sentence express anger?", ["Yes", "No"]}
]

reports =
  Enum.map(cases, fn {id, input, instructions, choices} ->
    rendered_prompt = ExLLama.Classifier.prompt(input, instructions, choices)

    {:ok, token_ids} =
      ExLLama.Nif.tokenize(classifier.model.resource, rendered_prompt, false)

    request = %GenAI.Classifier.Request{
      input: input,
      model: ExLLama.Classifier.model(classifier),
      questions: [
        %GenAI.Classifier.Question{id: id, instructions: instructions, choices: choices}
      ]
    }

    {:ok, result} = ExLLama.classify(classifier, request)
    [answer] = result.answers

    %{
      id: id,
      input: input,
      instructions: instructions,
      options: choices,
      rendered_prompt: rendered_prompt,
      token_ids: token_ids,
      index: answer.index,
      distribution: answer.distribution,
      selected_probability: answer.selected_probability,
      duration_us: result.duration
    }
  end)

{:ok, typed} =
  ExLLama.classify(classifier, %GenAI.Classifier.Request{
    input: "The meeting starts at 10 AM. It is now 9 AM.",
    model: ExLLama.Classifier.model(classifier),
    questions: [
      %GenAI.Classifier.Question{
        id: "started_boolean",
        type: :boolean,
        instructions: "Has the meeting started?"
      },
      %GenAI.Classifier.Question{
        id: "progress_score",
        type: :score,
        instructions: "Which level describes the meeting's progress?",
        rubric: [
          %{id: "pending", label: "Not started", score: 0},
          %{id: "active", label: "In progress", score: 10},
          %{id: "done", label: "Finished", score: 100}
        ]
      }
    ]
  })

true = typed.metadata.forward_passes == 2
{vendor_revision, 0} = System.cmd("git", ["-C", "c_src/llama.cpp", "rev-parse", "HEAD"])

File.write!(
  output_path,
  Jason.encode!(
    %{
      metadata:
        Map.merge(classifier.metadata, %{
          gpu_layers: gpu_layers,
          model_file: Path.basename(model_path),
          model_bytes: File.stat!(model_path).size,
          vendor_revision: String.trim(vendor_revision)
        }),
      cases: reports,
      typed_answers: Enum.map(typed.answers, &Map.from_struct/1),
      typed_duration_us: typed.duration
    },
    pretty: true
  )
)

IO.puts("Real native classifier validation saved: #{output_path}")
