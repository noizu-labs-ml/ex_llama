defmodule ExLLama.OutputQualityDiagTest do
  # Diagnostic lab for the output-quality epic (epic.llm-output-quality).
  #
  # Greedy decoding (top_k: 1) is deterministic: a sound pipeline
  # (tokenize -> decode -> sample -> detokenize) must produce identical
  # output across runs, and tinyllama's greedy continuation of a clean chat
  # prompt should be word-like text. Garbage tokens under GREEDY decoding
  # indicate a pipeline bug (sampling chain, context handling, backend), not
  # sampling-temperature variance. Each assertion prints its actual output
  # so the bisect verdict can be read straight from the CI log.
  use ExUnit.Case, async: false

  # Each test loads its own model + session and decodes on the CI CPU; the
  # runner is slow and tests in this module are serialized (async: false),
  # so give them headroom beyond the 60s default.
  @moduletag timeout: 180_000

  @model_path "local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  @prompt "<|user|>\n Say Hello. And only hello. Example \"Hello\".\n<|assistant|>\n Hello\n<|user|>\n Repeat what you just said.\n<|assistant|>\n Hello\n<|user|>\n Say Goodbye.\n<|assistant|>\n"

  defp priv_dir, do: :code.priv_dir(:ex_llama) |> List.to_string()

  defp greedy_completion(session, prompt, max_tokens) do
    # top_k: 1 pins the sampler to argmax => greedy/deterministic.
    ExLLama.Nif.completion(session.resource, prompt, %{
      max_tokens: max_tokens,
      seed: 2,
      temp: 0.0,
      top_p: 1.0,
      top_k: 1,
      min_p: 0.0
    })
  end

  test "greedy decoding is deterministic across repeated runs" do
    {:ok, llama} = ExLLama.load_model(priv_dir() <> "/models/" <> @model_path)
    {:ok, session} = ExLLama.create_session(llama)
    ExLLama.advance_context(session, @prompt)

    {:ok, %{content: run1}} = greedy_completion(session, Process.get({:ex_llama_ctx, session.resource}, @prompt), 32)
    {:ok, %{content: run2}} = greedy_completion(session, Process.get({:ex_llama_ctx, session.resource}, @prompt), 32)

    IO.puts("\n[diag] greedy run1: #{inspect(run1)}")
    IO.puts("[diag] greedy run2: #{inspect(run2)}")
    assert run1 == run2, "greedy decoding diverged between runs:\nrun1: #{inspect(run1)}\nrun2: #{inspect(run2)}"
  end

  test "greedy decoding produces word-like (non-garbage) output" do
    {:ok, llama} = ExLLama.load_model(priv_dir() <> "/models/" <> @model_path)
    {:ok, session} = ExLLama.create_session(llama)
    ExLLama.advance_context(session, @prompt)

    {:ok, %{content: out}} = greedy_completion(session, Process.get({:ex_llama_ctx, session.resource}, @prompt), 32)
    IO.puts("\n[diag] greedy output: #{inspect(out)}")

    # Garbage soup (e.g. "2,3,21:4 0 4 29.") has almost no multi-letter
    # alphabetic runs; coherent text has words of 3+ letters.
    word_runs = Regex.scan(~r/[A-Za-z]{3,}/, out) |> List.flatten()
    assert word_runs != [],
           "greedy output contains no alphabetic words of 3+ chars — pipeline " <>
             "produces garbage under deterministic decoding:\n#{inspect(out)}"
  end

  # ---- Round-2 probes (bisect the CI-only soup) ----
  #
  # Local repro of the exact NIF generation pattern (legacy llama_batch_get_one,
  # chunked ingest, same sampler chain/seed) against the same pin + GGUF is
  # CLEAN, so the remaining delta is CI-environment. Each probe toggles ONE
  # variable and prints its output; the verdict is read from the CI log.
  # Probes print only — the canonical word-heuristic assertion above stays
  # the epic oracle.

  defp new_session(overrides) do
    {:ok, llama} = ExLLama.load_model(priv_dir() <> "/models/" <> @model_path)
    {:ok, options} = ExLLama.Session.default_options()
    opts = Map.merge(Map.from_struct(options), Map.new(overrides))
    {:ok, session} = ExLLama.create_session(llama, opts)
    session
  end

  test "probe: greedy with flash attention disabled" do
    # LLAMA_FLASH_ATTN_TYPE_DISABLED = 0. If this is word-like while the
    # canonical greedy test (FA auto) is soup, the fault is in CPU FA kernels
    # on the runner.
    session = new_session(flash_attn_type: 0)
    ExLLama.advance_context(session, @prompt)

    {:ok, %{content: out}} = greedy_completion(session, @prompt, 32)
    words = Regex.scan(~r/[A-Za-z]{3,}/, out) |> List.flatten()
    IO.puts("\n[diag] greedy fa=disabled output: #{inspect(out)} (words: #{inspect(words)})")
  end

  test "probe: greedy with single thread" do
    # Rules out thread-parity / OpenMP reduction-order corruption on the
    # 4-thread default over a 2-4 vCPU runner.
    session = new_session(n_threads: 1, n_threads_batch: 1)
    ExLLama.advance_context(session, @prompt)

    {:ok, %{content: out}} = greedy_completion(session, @prompt, 32)
    words = Regex.scan(~r/[A-Za-z]{3,}/, out) |> List.flatten()
    IO.puts("\n[diag] greedy threads=1 output: #{inspect(out)} (words: #{inspect(words)})")
  end

  test "probe: default sampling chain (temp 0.8) seed 2" do
    # Context-level check: if the DEFAULT chain (top_k 40 / temp 0.8) with a
    # fixed seed also yields soup, the corruption is upstream of the sampler
    # (model decode / KV), not greedy-specific.
    session = new_session([])
    ExLLama.advance_context(session, @prompt)

    {:ok, %{content: out}} = ExLLama.Nif.completion(session.resource, @prompt, %{max_tokens: 32, seed: 2})
    words = Regex.scan(~r/[A-Za-z]{3,}/, out) |> List.flatten()
    IO.puts("\n[diag] default-chain seed=2 output: #{inspect(out)} (words: #{inspect(words)})")
  end
end
