defmodule ExLLama.KitchenSinkTest do
  @moduledoc """
  Kitchen-sink permutation sweep for session/completion defaults.

  Background: the shipped Elixir defaults struct populates every
  `ExLLama.SessionOptions` field, so `__model_nif_create_session__/2`
  forwards ALL of them — the "only forward explicitly-set keys" design is
  defeated. Several forwarded values deviate from llama.cpp's canonical
  defaults (n_threads/n_threads_batch 4 vs -1, n_batch 512 vs 2048,
  rope_scaling_type 0 = NONE vs -1 = UNSPECIFIED, yarn_* 0.0 vs -1.0
  inherit-sentinel). This suite empirically maps which option combinations
  produce coherent output on the current build, so the correct default set
  can be chosen from evidence rather than guesswork.

  Oracle: deterministic greedy completion (temp 0, top_k 1, fixed seed) is
  classified per run:
    :ok          non-empty, no tail cycle, no degenerate repetition
    :cycle       tail falls into a repeating cycle of length <= 4
                 (the CI failure signature: ", 0, 0, 0, ...")
    :degenerate  >= 12 tokens but < 20% unique tokens
    :garbage     < 70% printable ASCII in the detokenized output
    :empty       zero tokens generated
    :error       NIF returned {:error, _} (e.g. n_ctx guard, invalid type)
    :crash       raised/threw/exited (recorded, does not abort the sweep)

  Each combination runs against two scenarios:
    :short  ~20-token prompt  — exercises the decode path (ne11 == 1)
    :long   ~800-token prompt — forces chunked prefill (ne11 > 1), the
            batched-graph path that routes through BLAS/OpenMP batched matmul

  A golden reference is captured at suite start from the RECOMMENDED config
  (every forwardable key nil => pure llama.cpp raw defaults). Greedy outputs
  are compared to it (common-token-prefix ratio + token-set Jaccard) and the
  similarity is REPORTED, not asserted — floating-point reduction order
  legitimately varies across thread counts. Only the golden baseline itself
  must pass; if it does not, the build/environment is broken and every other
  row is untrustworthy.

  Usage (never runs under plain `mix test` — gated on KITCHEN_SINK env):

      KITCHEN_SINK=1 mix test test/ex_llama/kitchen_sink_test.exs --only kitchen_sink

      # phase selection: ofat | pairwise | random | all (default: all)
      KITCHEN_SINK=1 KITCHEN_SINK_PHASE=ofat mix test test/ex_llama/kitchen_sink_test.exs --only kitchen_sink

      # quick triage: long-prompt scenario only (the corruption-sensitive path)
      KITCHEN_SINK=1 KITCHEN_SINK_SCENARIO=long KITCHEN_SINK_PHASE=ofat mix test ...

      # knobs: KITCHEN_SINK_LIMIT=<n>  cap total combos
      #        KITCHEN_SINK_RANDOM_N=<n> random-phase sample count (default 48)
      #        KITCHEN_SINK_RESULTS=<path> JSONL output (default: system tmp)
      #        KITCHEN_SINK_MODEL=<abs path> model override

  Env-var toggles the NIF reads at load time (GGML_CPU_DISABLE_FUSION, etc.)
  cannot vary inside one VM run — re-run the whole suite with the env var set
  to compare. Build-flag variants (GGML_NATIVE=OFF, BLAS on/off) likewise
  need their own compiled build + suite run.
  """

  use ExUnit.Case, async: false

  @moduletag :kitchen_sink
  # Gate: plain `mix test` must not run a multi-minute sweep.
  unless System.get_env("KITCHEN_SINK") do
    @moduletag :skip
  end

  @moduletag timeout: :infinity

  @seed 42
  @max_tokens 24

  @model_rel "models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

  # Every key the NIF layer can forward (see ExLLama.Nif forward_keys +
  # embeddings/offload_kqv). nil => NOT forwarded => llama.cpp raw default.
  @forwardable [
    :n_ctx, :n_batch, :n_ubatch, :n_seq_max,
    :n_threads, :n_threads_batch,
    :rope_scaling_type, :rope_freq_base, :rope_freq_scale,
    :yarn_ext_factor, :yarn_attn_factor, :yarn_beta_fast, :yarn_beta_slow,
    :yarn_orig_ctx,
    :type_k, :type_v,
    :flash_attn_type, :pooling_type, :attention_type,
    :op_offload, :kv_unified, :swa_full,
    :embedding, :offload_kqv
  ]

  # ---------------------------------------------------------------------------
  # Setup
  # ---------------------------------------------------------------------------

  setup_all do
    System.put_env("LLAMA_LOG_DISABLE", "1")

    model_path =
      System.get_env("KITCHEN_SINK_MODEL") ||
        Enum.find([Path.join(File.cwd!(), "priv"), :code.priv_dir(:ex_llama) |> List.to_string()], fn base ->
          base != nil and File.exists?(Path.join(base, @model_rel))
        end)
        |> case do
          nil -> Path.join([File.cwd!(), "priv", @model_rel])
          base -> Path.join(base, @model_rel)
        end

    unless File.exists?(model_path) do
      flunk("model not found at #{model_path} — run priv/models init or set KITCHEN_SINK_MODEL")
    end

    {:ok, model} = ExLLama.load_model(model_path)

    # Golden references from the RECOMMENDED config: nothing forwarded except
    # n_ctx (a capacity setting, orthogonal to numerics — raw default 512
    # cannot hold the long prompt, as the first smoke run confirmed).
    golden_opts = recommended_opts(n_ctx: 2048)
    golden_short = golden_probe(model, :short, golden_opts)
    golden_long = golden_probe(model, :long, golden_opts)

    for {sc, g} <- [short: golden_short, long: golden_long] do
      unless g.status in [:ok, :empty] do
        flunk(
          "GOLDEN BASELINE FAILED (#{sc}): recommended/raw-default config produced " <>
            "status=#{g.status} detail=#{inspect(g.detail)} — build or environment " <>
            "is broken; permutation results are untrustworthy."
        )
      end
    end

    IO.puts("""
    [kitchen-sink] environment metadata:
      model: #{model_path}
      cpu flags: #{inspect(cpu_flags())}
      n_schedulers: #{inspect(:erlang.system_info(:schedulers_online))}
      os: #{inspect(:os.type())}
    [kitchen-sink] golden(short/#{golden_short.status}): #{preview(golden_short.content)}
    [kitchen-sink] golden(long/#{golden_long.status}): #{preview(golden_long.content)}
    """)

    %{model: model, golden: %{short: golden_short, long: golden_long}}
  end

  # ---------------------------------------------------------------------------
  # The sweep
  # ---------------------------------------------------------------------------

  @tag :kitchen_sink
  test "kitchen sink permutation sweep", %{model: model, golden: golden} do
    phase = (System.get_env("KITCHEN_SINK_PHASE") || "all") |> String.to_atom()
    limit = env_int("KITCHEN_SINK_LIMIT", :infinity)
    scenarios = scenarios()

    combos =
      [
        {:baseline,
         [
           {"shipped-defaults", shipped_opts()},
           # The pre-fix production config (explicit 4/4 threads, rope NONE,
           # yarn 0.0 overrides) — the suspected CI-garbage recipe.
           {"legacy-defaults (pre-fix)",
            recommended_opts(
              n_ctx: 2048, n_batch: 512, n_threads: 4, n_threads_batch: 4,
              rope_scaling_type: 0,
              yarn_ext_factor: 0.0, yarn_attn_factor: 0.0,
              yarn_beta_fast: 0.0, yarn_beta_slow: 0.0,
              type_k: 1, type_v: 1, embedding: false, offload_kqv: true
            )}
         ]},
        {:ofat, ofat_combos()},
        {:pairwise, pairwise_combos()},
        {:random, random_combos(env_int("KITCHEN_SINK_RANDOM_N", 48))}
      ]
      |> Enum.filter(fn {p, _} -> phase == :all or p == phase end)
      |> Enum.flat_map(fn {p, list} -> Enum.map(list, fn {label, opts} -> {p, label, opts} end) end)
      |> cap_combos(limit)

    total = length(combos) * length(scenarios)
    IO.puts(:stderr, "[kitchen-sink] phase=#{phase} combos=#{length(combos)} rows=#{total} (limit=#{limit})")

    # Stream rows to JSONL as they complete — a mid-run crash keeps partials.
    results_path = results_path()

    rows =
      for {{p, label, opts}, i} <- Enum.with_index(combos),
          sc <- scenarios do
        row = run_row(model, golden, p, label, opts, sc)
        File.write!(results_path, json(row) <> "\n", [:append])

        if rem(i + 1, 10) == 0 do
          IO.puts(:stderr, "[kitchen-sink] #{(i + 1) * length(scenarios)}/#{total}...")
        end

        report_row(row)

        # Sessions hold multi-MB KV caches as NIF resources; reclaim promptly
        # so a 200-run sweep does not accumulate GBs.
        :erlang.garbage_collect()
        row
      end

    write_results(rows, results_path)
    summarize(rows)

    # Only structural sanity assertions: the sweep's value is the printed /
    # persisted matrix, not pass/fail per combination.
    assert rows != []
    assert Enum.any?(rows, &(&1.status == :ok))
  end

  # -- combination spaces ----------------------------------------------------

  defp shipped_opts do
    # The shipped Elixir default struct — every field populated, everything
    # forwarded. This is what production code gets today.
    {:ok, defaults} = ExLLama.Session.default_options()
    Map.from_struct(defaults)
  end

  defp recommended_opts(overrides \\ []) do
    # All forwardable keys nil (=> raw llama.cpp defaults) except overrides.
    @forwardable
    |> Map.new(fn k -> {k, nil} end)
    |> Map.merge(Map.new(overrides))
    |> Map.put(:seed, @seed)
  end

  # One-factor-at-a-time probes against the raw-default baseline.
  defp ofat_combos do
    dims = [
      n_threads: [-1, 0, 1, 2, 4, 8],
      n_threads_batch: [-1, 0, 1, 2, 4, 8],
      n_batch: [32, 64, 512, 2048],
      n_ubatch: [32, 512, 2048],
      n_ctx: [0, 512, 2048],
      rope_scaling_type: [-1, 0, 1],
      rope_freq_base: [0.0, 10_000.0],
      rope_freq_scale: [0.0, 1.0],
      yarn_ext_factor: [-1.0, 0.0, 1.0],
      yarn_attn_factor: [0.0, 1.0],
      yarn_beta_fast: [-1.0, 32.0],
      yarn_beta_slow: [-1.0, 1.0],
      yarn_orig_ctx: [0, 2048],
      type_k: [0, 1, 8],
      type_v: [0, 1, 8],
      flash_attn_type: [-1, 0, 1],
      n_seq_max: [1, 2],
      kv_unified: [true, false],
      swa_full: [true, false],
      offload_kqv: [true, false],
      op_offload: [true, false]
    ]

    ofat =
      for {k, vals} <- dims, v <- vals do
        {"#{k}=#{inspect(v)}", recommended_opts([{k, v}])}
      end

    # The whole shipped-defaults yarn group as one probe too (that is how it
    # is actually forwarded in production).
    yarn_group = [
      yarn_ext_factor: 0.0, yarn_attn_factor: 0.0,
      yarn_beta_fast: 0.0, yarn_beta_slow: 0.0
    ]

    [{"yarn-group all 0.0 (shipped)", recommended_opts(yarn_group)} | ofat]
  end

  # Pairwise grids over the corruption-sensitive factors; the long-prompt
  # scenario is where these matter (prefill ne11 > 1 -> BLAS/batched graph).
  defp pairwise_combos do
    nt_x_ntb =
      for nt <- [-1, 0, 1, 2, 4, 8], ntb <- [-1, 0, 1, 2, 4, 8] do
        {"nt=#{nt} ntb=#{ntb}", recommended_opts(n_threads: nt, n_threads_batch: ntb)}
      end

    ntb_x_batch =
      for ntb <- [-1, 1, 4], nb <- [32, 512, 2048] do
        {"ntb=#{ntb} n_batch=#{nb}", recommended_opts(n_threads_batch: ntb, n_batch: nb)}
      end

    ntb_x_fa =
      for ntb <- [-1, 1, 4], fa <- [-1, 0, 1] do
        {"ntb=#{ntb} fa=#{fa}", recommended_opts(n_threads_batch: ntb, flash_attn_type: fa)}
      end

    fa_x_kv =
      for fa <- [-1, 0, 1], tk <- [0, 1, 8] do
        {"fa=#{fa} type_k=#{tk}", recommended_opts(flash_attn_type: fa, type_k: tk, type_v: tk)}
      end

    nt_x_ntb ++ ntb_x_batch ++ ntb_x_fa ++ fa_x_kv
  end

  # Deterministic random sampling of the joint space (60% of dims included
  # per draw). n_ubatch is clamped to <= n_batch where both are set, because
  # upstream treats n_ubatch > n_batch as invalid.
  defp random_combos(n) do
    :rand.seed(:exsss, {@seed, @seed, @seed})

    dims = [
      {:n_threads, [-1, 0, 1, 2, 4, 8]},
      {:n_threads_batch, [-1, 0, 1, 2, 4, 8]},
      {:n_batch, [32, 64, 512, 2048]},
      {:n_ubatch, [32, 512, 2048]},
      {:n_ctx, [0, 512, 2048]},
      {:rope_scaling_type, [-1, 0, 1]},
      {:rope_freq_base, [0.0, 10_000.0]},
      {:rope_freq_scale, [0.0, 1.0]},
      {:yarn_ext_factor, [-1.0, 0.0, 1.0]},
      {:type_k, [0, 1, 8]},
      {:type_v, [0, 1, 8]},
      {:flash_attn_type, [-1, 0, 1]}
    ]

    for i <- 1..n do
      chosen =
        dims
        |> Enum.filter(fn _ -> :rand.uniform() < 0.6 end)
        |> Enum.map(fn {k, vals} -> {k, Enum.random(vals)} end)
        |> clamp_ubatch()

      {"rand##{i} #{kw_inspect(chosen)}", recommended_opts(chosen)}
    end
  end

  defp clamp_ubatch(chosen) do
    case {Keyword.get(chosen, :n_ubatch), Keyword.get(chosen, :n_batch)} do
      {ub, b} when is_integer(ub) and is_integer(b) and ub > b ->
        Keyword.put(chosen, :n_batch, ub)

      _ ->
        chosen
    end
  end

  # -- execution --------------------------------------------------------------

  defp run_row(model, golden, phase, label, opts, scenario) do
    t0 = System.monotonic_time(:millisecond)

    outcome =
      try do
        run_probe(model, opts, scenario)
      rescue
        e -> {:crash, Exception.message(e)}
      catch
        :exit, reason -> {:crash, "exit: #{inspect(reason)}"}
        kind, value -> {:crash, "#{inspect(kind)}: #{inspect(value)}"}
      end

    dt = System.monotonic_time(:millisecond) - t0

    case outcome do
      {:error, reason} ->
        row(phase, label, opts, scenario, nil, :error, 0, "", reason, dt)

      {:crash, msg} ->
        row(phase, label, opts, scenario, nil, :crash, 0, "", msg, dt)

      {:ok, tokens, content, ctx_size, n} ->
        {status, detail} = classify_content(content, tokens, n)

        sim =
          case golden[scenario] do
            %{status: :ok, tokens: g_tokens} when g_tokens != [] and tokens != [] ->
              "prefix=#{prefix_similarity(tokens, g_tokens)} jac=#{jaccard(tokens, g_tokens)}"

            _ ->
              nil
          end

        detail =
          [detail, sim]
          |> Enum.reject(&is_nil/1)
          |> Enum.join(" ")
          |> case do
            "" -> nil
            x -> x
          end

        row(phase, label, opts, scenario, ctx_size, status, length(tokens), content, detail, dt)
    end
  end

  # Returns:
  #   {:ok, tokens, content, ctx_size, token_count}
  #   {:error, reason}
  defp run_probe(model, opts, scenario) do
    # The long prompt (~800 tokens) cannot fit the raw-default n_ctx of 512;
    # pin capacity on the long scenario unless the probe is about n_ctx itself
    # (same treatment as the golden baseline — capacity, not numerics).
    opts =
      if scenario == :long and is_nil(opts[:n_ctx]) do
        Map.put(opts, :n_ctx, 2048)
      else
        opts
      end

    prompt = prompt_for(scenario)

    case ExLLama.create_session(model, opts) do
      {:error, reason} ->
        {:error, to_string(reason)}

      {:ok, session} ->
        {:ok, ctx_size} = ExLLama.Session.context_size(session)

        greedy = %{max_tokens: @max_tokens, seed: @seed, temp: 0.0, top_k: 1}

        case ExLLama.Nif.completion(session.resource, prompt, greedy) do
          {:error, reason} ->
            {:error, to_string(reason)}

          {:ok, %{content: content, token_length: n}} ->
            {:ok, tokenize(model, content), content, ctx_size, n}
        end
    end
  end

  defp golden_probe(model, scenario, opts) do
    case run_probe(model, opts, scenario) do
      {:ok, tokens, content, _ctx_size, n} ->
        {status, detail} = classify_content(content, tokens, n)
        %{status: status, detail: detail, tokens: tokens, content: content}

      {:error, reason} ->
        %{status: :error, detail: reason, tokens: [], content: nil}
    end
  end

  defp classify_content(content, _tokens, n) do
    ids = tokenize_cached(content)

    cond do
      n == 0 or content == "" ->
        {:empty, "0 tokens"}

      n >= 12 and unique_ratio(ids) < 0.2 ->
        {:degenerate, "unique=#{Float.round(unique_ratio(ids), 3)}"}

      n >= 12 and tail_cycle(ids) > 0 ->
        {:cycle, "cycle_len=#{tail_cycle(ids)}"}

      printable_ratio(content) < 0.7 ->
        {:garbage, "printable=#{Float.round(printable_ratio(content), 2)}"}

      true ->
        {:ok, nil}
    end
  end

  # -- metrics -----------------------------------------------------------------

  defp tokenize(model, content) do
    case ExLLama.Nif.tokenize(model.resource, content, false) do
      {:ok, ids} -> ids
      _ -> []
    end
  end

  # classify_content has no model handle; approximate tokenization by
  # whitespace so classification still works without the model in scope.
  defp tokenize_cached(content), do: String.split(content, ~r/\s+/, trim: true)

  defp unique_ratio([]), do: 1.0
  defp unique_ratio(ids), do: length(Enum.uniq(ids)) / length(ids)

  # Smallest period l <= 4 such that the last (min 3*l, 12) tokens repeat with
  # period l — the ", 0, 0, 0" CI signature is l in 1..2.
  defp tail_cycle(ids, max_cycle \\ 4, window \\ 12) do
    w = min(length(ids), window)
    tail = Enum.take(ids, -w)
    max_l = min(max_cycle, div(w, 3))

    if max_l >= 1 do
      Enum.find(1..max_l, fn l -> w >= 3 * l and periodic?(tail, l) end) || 0
    else
      0
    end
  end

  defp periodic?(tail, l) do
    tail
    |> Enum.with_index()
    |> Enum.drop(l)
    |> Enum.all?(fn {v, i} -> v == Enum.at(tail, i - l) end)
  end

  defp printable_ratio(content) do
    chars = String.to_charlist(content)
    printable = Enum.count(chars, fn c -> c in 32..126 or c == ?\n or c == ?\t end)
    if chars == [], do: 1.0, else: printable / length(chars)
  end

  defp prefix_similarity(ids, golden_ids) do
    common = ids |> Enum.zip(golden_ids) |> Enum.take_while(fn {a, b} -> a == b end) |> length()
    max_l = max(length(ids), length(golden_ids))
    if max_l == 0, do: 1.0, else: Float.round(common / max_l, 3)
  end

  defp jaccard(ids, golden_ids) do
    a = MapSet.new(ids)
    b = MapSet.new(golden_ids)
    union = MapSet.size(MapSet.union(a, b))
    inter = MapSet.size(MapSet.intersection(a, b))
    if union == 0, do: 1.0, else: Float.round(inter / union, 3)
  end

  # -- reporting ------------------------------------------------------------------

  defp row(phase, label, opts, scenario, ctx_size, status, n_tokens, content, detail, dt) do
    %{
      phase: phase,
      label: label,
      scenario: scenario,
      status: status,
      ctx_size: ctx_size,
      n_tokens: n_tokens,
      content: content,
      detail: detail,
      ms: dt,
      opts: Map.drop(opts, [:seed])
    }
  end

  defp report_row(row) do
    # :stderr — ExUnit captures stdout until the test finishes; stderr streams
    # live so a long sweep shows progress and survives a mid-run crash.
    IO.puts(:stderr, row_line(row))
  end

  defp row_line(row) do
    "  [#{row.phase}|#{row.scenario}] #{row.status} ctx=#{row.ctx_size} " <>
      "tok=#{row.n_tokens} ms=#{row.ms} :: #{row.label}" <>
      if(row.detail, do: " (#{row.detail})", else: "") <>
      if(row.status in [:ok, :cycle, :degenerate, :garbage] and is_binary(row.content),
        do: " \"#{preview(row.content)}\"",
        else: ""
      )
  end

  defp preview(nil), do: ""
  defp preview(s), do: s |> String.replace("\n", "\\n") |> String.slice(0, 60)

  defp summarize(rows) do
    by_status = Enum.frequencies_by(rows, & &1.status)
    bad = Enum.filter(rows, &(&1.status not in [:ok, :empty]))

    IO.puts("""
    [kitchen-sink] SUMMARY
      status counts: #{inspect(by_status)}
      failing/interesting rows: #{length(bad)}
    """)

    Enum.each(bad, fn r ->
      IO.puts("    #{r.status} :: #{r.phase}|#{r.scenario} :: #{r.label} :: #{r.detail}")
    end)

    # Thread matrix for the pairwise phase (the Mac BLAS corruption map).
    matrix =
      rows
      |> Enum.filter(&String.starts_with?(&1.label, "nt="))
      |> Enum.map(&{&1.label, &1.scenario, &1.status})

    if matrix != [] do
      IO.puts("  thread matrix (nt/ntb -> status per scenario):")

      matrix
      |> Enum.group_by(fn {label, _, _} -> label end, fn {_, sc, st} -> {sc, st} end)
      |> Enum.sort()
      |> Enum.each(fn {label, scs} ->
        IO.puts("    #{label} :: #{Enum.map_join(scs, ", ", fn {sc, st} -> "#{sc}=#{st}" end)}")
      end)
    end
  end

  defp results_path do
    default = Path.join(System.tmp_dir!(), "kitchen-sink-#{System.system_time(:second)}.jsonl")
    System.get_env("KITCHEN_SINK_RESULTS") || default
  end

  defp write_results(_rows, path) do
    IO.puts("[kitchen-sink] results written to #{path}")
  end

  # Minimal JSON encoder (diagnostics file: numbers, strings, bools, atoms,
  # lists, flat maps only — no dependency on Jason).
  defp json(nil), do: "null"
  defp json(true), do: "true"
  defp json(false), do: "false"
  defp json(a) when is_atom(a), do: json(Atom.to_string(a))
  defp json(i) when is_integer(i), do: Integer.to_string(i)
  defp json(f) when is_float(f), do: Float.to_string(f)
  defp json(b) when is_binary(b), do: inspect(b)
  defp json(l) when is_list(l), do: "[" <> Enum.map_join(l, ",", &json/1) <> "]"

  defp json(m) when is_map(m) do
    "{" <> Enum.map_join(m, ",", fn {k, v} -> "\"#{k}\":" <> json(v) end) <> "}"
  end

  # -- fixtures --------------------------------------------------------------------

  defp prompt_for(:short) do
    "<|user|>\n Say Hello. And only hello. Example \"Hello\".\n<|assistant|>\n"
  end

  defp prompt_for(:long) do
    # ~800 tokens: 40 QA pairs. Crosses the 512/2048 n_batch/n_ubatch chunk
    # boundaries so the batched prefill path (ne11 > 1) is exercised.
    1..40
    |> Enum.map_join(fn i -> "<|user|>\n Say the number #{i}.\n<|assistant|>\n #{i}\n" end)
    |> Kernel.<>("<|user|>\n Now say Goodbye.\n<|assistant|>\n")
  end

  defp scenarios do
    case System.get_env("KITCHEN_SINK_SCENARIO") do
      "short" -> [:short]
      "long" -> [:long]
      _ -> [:short, :long]
    end
  end

  defp env_int(name, default) do
    case Integer.parse(System.get_env(name) || "") do
      {v, _} -> v
      :error -> default
    end
  end

  defp cap_combos(combos, :infinity), do: combos
  defp cap_combos(combos, n) when is_integer(n) and n > 0, do: Enum.take(combos, n)

  defp kw_inspect(kw), do: Enum.map_join(kw, " ", fn {k, v} -> "#{k}=#{inspect(v)}" end)

  defp cpu_flags do
    cond do
      File.exists?("/proc/cpuinfo") ->
        case File.read("/proc/cpuinfo") do
          {:ok, txt} ->
            txt
            |> String.split("\n")
            |> Enum.find(&String.starts_with?(&1, "flags"))
            |> case do
              nil ->
                :no_flags_line

              line ->
                line |> String.split() |> Enum.filter(&(&1 in ["avx512_fp16", "avx512f", "avx2", "f16c"]))
            end

          _ ->
            :unreadable
        end

      true ->
        case System.cmd("sysctl", ["-n", "machdep.cpu.features"], stderr_to_stdout: true) do
          {out, 0} -> String.slice(out, 0, 120)
          _ -> :unknown
        end
    end
  rescue
    _ -> :unknown
  end
end
