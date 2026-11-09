# Native Jev classifier

<!-- 〚📁:ebd58437-a823-594a-8fb5-9694f9b8b356〛 docs/jev-classifier.md :: Native Jev-Omni classifier setup and validation -->

ExLLama's classifier uses its existing llama.cpp C++ backend. It performs text
inference with Jev-Omni's final-token hidden state and separate FP32 decision
head. It does not sample generated tokens. The classifier preserves ordered
choices, boolean `[false, true]` probabilities, and numeric rubric levels.

`akhilaaa3/Jev-Omni` is independent of TypeSafe's hosted Jev. Hosted APIs belong
in GenAI provider adapters. Image/audio/video inputs are explicitly rejected by
this implementation.

## API

```elixir
{:ok, classifier} = ExLLama.load_classifier(
  "/models/Jev-Omni-Q8_0.gguf", "/models/decision-head.bin",
  model_id: "akhilaaa3/Jev-Omni",
  model_options: [n_gpu_layers: 99],
  context_options: [n_ctx: 2048, n_batch: 256, n_ubatch: 256]
)

request = %GenAI.Classifier.Request{
  input: "The meeting starts at 10 AM. It is now 9 AM.",
  model: ExLLama.Classifier.model(classifier),
  questions: [
    %GenAI.Classifier.Question{
      id: "started", type: :boolean,
      instructions: "Has the meeting started?"
    },
    %GenAI.Classifier.Question{
      id: "route", type: :choice,
      instructions: "Which action best fits this state?",
      choices: [%{id: "wait", label: "Wait for the meeting"},
                %{id: "join", label: "Join an ongoing meeting"}]
    }
  ]
}

{:ok, result} = ExLLama.classify(classifier, request)
# Equivalent explicit Core routing:
{:ok, result} = GenAI.classify(request,
  provider: ExLLama.Classifier, classifier: classifier)
```

Load a backbone/head once. Each question runs a separate forward pass and owns
a private native context, freed on success and failure. A model resource mutex
serializes its classifier calls. This does not replace or reuse a generation
session's KV state. The native feature vector uses `llama_model_n_embd_out`,
pooling `NONE`, embedding mode and the last output token, without L2
normalization. Head width must match the output width before classification.

Question defaults describe a boolean judgment, best matching choice, or rubric
level. Supply `instructions` for application judgments. Choice IDs remain
separate from human-readable labels. Score rubrics use ordered unique numeric
scores; `expected_score` is their weighted mean, while `grade` identifies the
winning level. Ties select the first option. Native confidence is explicitly
the maximum option probability, with metadata identifying that meaning.

The adapter accepts 2–256 options. The upstream model's established quality is
primarily for at most 20 options. It reports sequential question execution in
metadata. A supplied Core model ID must match the loaded classifier's
`model_id`. The default ID is the GGUF path.

## Head format and export

`JEVH0001` contains an eight-byte magic, little-endian uint32 hidden width and
class count, then little-endian FP32 `mu`, `sd`, row-major `linear.weight` and
`linear.bias`. The class count is exactly 256. Dimensions, exact file length,
finite parameters and positive standard deviations are checked before use.
The head computes `(hidden - mu) / sd`, then the linear projection, followed
by a stable softmax over only the supplied option slots.

```sh
python3 scripts/export_jev_head.py head.pt decision-head.bin
# Or the publisher's non-executable FP32 archive:
python3 scripts/export_jev_head.py decision-head-f32.npz decision-head.bin
```

The exporter requires NumPy; PT input additionally requires PyTorch and uses
`weights_only=True`. NPZ input uses `allow_pickle=False`. The export prints
only its output checksum. Weights are not packaged in this library.

## Build and validation prerequisites

The feature checkout is based on updated ExLLama `main` at
`a131d1c3c8e3133b762a79484fa1b41f4fb61b08`, per the user's explicit base
change, and pins llama.cpp
`d81235049384534c167caea52b85a694f6103d14` (v0.6.0). The pinned header declares
`llama_model_n_embd_out`. Initialize that vendor only after the required Git
metadata/network approval is available:

```sh
git submodule update --init --depth 1 c_src/llama.cpp
```

Required native tools: C++17 compiler, CMake, Erlang headers and the pinned
llama.cpp source. CUDA validation additionally needs the matching CUDA toolkit
and GPU backend build. The Makefile's default macOS build uses Metal. Keep
`deps` and `_build` in the canonical ExLLama checkout with absolute symlinks
from the feature worktree; resolve dependencies using the feature's `mix.exs`.
The default Core dependency is pinned to
merged Core commit `a98208599ab003f6ef1cc8ffdb01e91d01d5331b` (classifier API 0.4,
including required JSON and YAML serializers). Set
`GENAI_CORE_PATH` only when developing against another local Core checkout.
Resolve dependencies using this feature's C++ dependency metadata.

```sh
env ASDF_ERLANG_VERSION=28.5 ASDF_ELIXIR_VERSION=1.19.5-otp-28 mix deps.get
env ASDF_ERLANG_VERSION=28.5 ASDF_ELIXIR_VERSION=1.19.5-otp-28 mix test
make classifier-head-test
make test
python3 scripts/test_export_jev_head.py
```

## Live validation

The documented `noizu.server` has two Tesla V100 16 GB GPUs and approximately
504 GiB RAM. A 36 GiB Apple Silicon host is also available for local validation.
Inspect current free memory before loading. Prefer Q8_0 (~12.67 GB) if runtime
and context fit; Q4_K_M (~7.38 GB) is an alternative with its quantization
recorded in the validation report. Start with 2,048 context and 256-token batches.

The [published GGUF package](https://huggingface.co/ngquocvinh/Jev-Omni-GGUF)
provides its FP32 head and a reference adapter. Its
[manifest](https://huggingface.co/ngquocvinh/Jev-Omni-GGUF/blob/f07b38e109cbc9b338029f88faeb3e6b6f894c73/reproducibility/manifest.md)
records the source revision and hashes, including bit-for-bit verification
against upstream `head.pt`. Text inference does not require the multimodal
projector.

The validated publisher revision is `f07b38e109cbc9b338029f88faeb3e6b6f894c73`.
The Q8 GGUF SHA256 is
`9eb606c2a26c9125e17b1a947aff2890dcc19a2525985bdca416ed8f301a0519`;
the exported portable head SHA256 is
`1de60fa1515b27884ed1c0b59c782681e1022519fe4ee180c251699dc3a20187`.

After vendor/build approval and model provisioning:

```sh
env ASDF_ERLANG_VERSION=28.5 ASDF_ELIXIR_VERSION=1.19.5-otp-28 \
  mix run scripts/validate_jev_classifier.exs \
  /models/Jev-Omni-Q8_0.gguf /models/decision-head.bin /tmp/jev-native.json
```

Run the publisher's reference adapter against the identical GGUF/head and the
same three cases in the script, with pooling `none` and unnormalized
embeddings. Compare choice identity and each probability, recording maximum
absolute probability drift and both runtimes. Native FP32 accumulation and
the reference's BLAS reduction may differ slightly; report observed drift
before choosing an acceptance threshold. Comparing the same quantization
isolates adapter drift; it does not measure BF16-to-quantized model quality.

For the upstream meeting case, the reference and comparison commands are:

```sh
python3 jev_omni_gguf_decide.py --head /models/decision-head-f32.npz \
  --server http://127.0.0.1:8080 \
  --state 'The meeting starts at 10 AM. It is now 9 AM.' \
  --question 'Has the meeting started?' --options Yes No > /tmp/jev-reference-meeting.json
python3 scripts/compare_jev_parity.py /tmp/jev-native.json \
  --reference meeting=/tmp/jev-reference-meeting.json
```

The reference server must already have been built and started with the same
GGUF, `--embedding --pooling none --embd-normalize -1`, modest context and GPU
offload settings. These commands provision nothing and do not launch a service.

To build the reference server from the same pinned source, finish the NIF build
first, then configure and build the server target in the existing CMake tree:

```sh
cmake -S c_src/llama.cpp -B c_src/llama.cpp/build \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build c_src/llama.cpp/build --target llama-server -j8
c_src/llama.cpp/build/bin/llama-server \
  -m /models/Jev-Omni-Q8_0.gguf --host 127.0.0.1 --port 18082 \
  -c 2048 -b 256 -ub 256 -ngl 99 --parallel 1 \
  --embedding --pooling none --embd-normalize -1
```

Run the server after the native validation process exits, so two processes do
not allocate the large model simultaneously. The following runner imports the
reviewed publisher adapter, repeats all three cases from the native report, and
saves each response plus a combined `parity.json` with token IDs, choice agreement,
probability drift and both per-call durations:

```sh
python3 scripts/run_jev_reference.py \
  --adapter /models/jev_omni_gguf_decide.py --native /tmp/jev-native.json \
  --head /models/decision-head-f32.npz --server http://127.0.0.1:18082 \
  --output-directory /tmp/jev-reference
```

The native report records its source revision, model byte count and trained head
checksum. Its explicit BOS token must match the server's automatically added BOS;
the runner checks complete token sequences before comparing inference results.
The validation script also exercises a Boolean question and a nonlinear score
rubric in one multiquestion request. Set `JEV_GPU_LAYERS=0` for CPU-only runs.

## Verified acceptance

The portable C++ head test, twelve classifier contract/native tests, 67 existing
generation/template tests, fourteen standalone C++ regressions and three NPZ
exporter tests ran successfully. The actual trained 3,840-by-256 head also
matched NumPy computation for 2, 3, 20 and 256 options using deterministic
features, with maximum probability drift `1.3543104e-7` and identical choices.
That comparison validates head math. Real native Q8 model inference also passed
the three choice cases, a Boolean question and a nonlinear score rubric. The
meeting, refund and neutral-sentiment cases selected the expected answers, with
native per-call durations of approximately 1.008, 0.403 and 0.623 seconds on the
36 GiB Apple Silicon validation host, using Metal, context 2,048 and batch 256.

The reviewed publisher adapter ran against the same pinned model/head and
llama.cpp server. All three choices and complete token sequences matched:

| Case | Tokens | Maximum absolute probability drift | Native / reference seconds |
| --- | ---: | ---: | ---: |
| Meeting | 74 | `1.7737e-8` | 1.008 / 0.415 |
| Refund | 74 | `5.6984e-8` | 0.403 / 0.387 |
| Neutral anger | 64 | `1.5017e-8` | 0.623 / 0.289 |

These are small acceptance examples and same-quantization adapter comparisons;
they do not establish model accuracy, BF16 parity or a general speed benchmark.
