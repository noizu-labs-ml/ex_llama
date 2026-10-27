# 〚📁:af67a3ff-66e3-5cce-8861-ed959e16f39e〛 mix.exs :: Mix project config, deps and native build settings
defmodule ExLLama.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_llama,
      name: "LLama CPP Nif Wrapper",
      description: description(),
      package: package(),
      version: "0.4.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      # Worktrees share canonical caches. Resolve their links before Mix computes
      # priv symlinks, so native libraries and model fixtures remain reachable.
      deps_path: cache_path("deps"),
      build_path: cache_path("_build"),
      docs: [
        main: "ExLLama",
        extras: [
          "README.md",
          "docs/jev-classifier.md",
          "LICENSE"
        ]
      ],
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/project.plt"}
      ],
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps()
    ]
  end

  defp cache_path(name) do
    case File.read_link(Path.join(__DIR__, name)) do
      {:ok, target} -> Path.expand(target, __DIR__)
      _ -> name
    end
  end

  defp description() do
    "NIF Wrapper around llama.cpp allowing elixir code to load/infer against gguf format models."
  end

  defp package() do
    [
      licenses: ["MIT"],
      links: %{
        project: "https://github.com/noizu-labs-ml/ex_llama",
        developer_github: "https://github.com/noizu"
      },
      files:
        ~w(lib c_src/ex_llama_nif.cpp c_src/jev_head.h c_src/test_llama_nif.cpp c_src/test_jev_head.cpp scripts/*.py scripts/*.exs docs/jev-classifier.md Makefile mix.exs README.md CHANGELOG.md LICENSE*)
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.9", runtime: false},
      # Documentation Provider
      {:ex_doc, "~> 0.40", only: [:dev, :test], optional: true, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      genai_core_dep(),
      {:finch, "~> 0.15", optional: true},
      {:elixir_uuid, "~> 1.2", optional: true},
      {:shortuuid, "~> 4.0", optional: true}

      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end

  defp genai_core_dep do
    case System.get_env("GENAI_CORE_PATH") do
      nil ->
        {:genai_core,
         git: "https://github.com/noizu-labs-ml/genai-core.git",
         ref: "a98208599ab003f6ef1cc8ffdb01e91d01d5331b"}

      path ->
        {:genai_core, path: path, override: true}
    end
  end
end
