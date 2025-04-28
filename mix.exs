defmodule ExLLama.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_llama,
      name: "LLama CPP Nif Wrapper",
      description: description(),
      package: package(),
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      rustler_crates: rustler_crates(),
      docs: [
        main: "ExLLama",
        extras: [
          "README.md",
          "LICENSE"
        ]
      ],
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/project.plt"}
      ],
      deps: deps()
    ]
  end


  defp description() do
    "NIF Wrapper around the rust LLamaCPP client allowing elixir code to load/infer against gguf format models."
  end

  defp rustler_crates do
    [
      erlang_llama_cpp_nif: [
        path: "native/erlang_llama_cpp_nif",
        mode: rustc_mode(Mix.env())
      ]
    ]
  end
  
  defp rustc_mode(:prod), do: :release
  defp rustc_mode(_), do: :debug
  
  defp package() do
    [
      licenses: ["MIT"],
      links: %{
        project: "https://github.com/noizu-labs-ml/ex_llama",
        developer_github: "https://github.com/noizu"
      },
      files: ~w(lib native priv mix.exs README.md CHANGELOG.md LICENSE*),
      exclude_patterns: ["priv/models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"]
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
      {:rustler, "~> 0.32.1", runtime: false},
      {:ex_doc, "~> 0.28.3", only: [:dev, :test], optional: true, runtime: false}, # Documentation Provider
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:genai_core, "~> 0.2"},
      {:finch, "~> 0.15", optional: true},
      {:elixir_uuid, "~> 1.2", optional: true},
      {:shortuuid, "~> 3.0", optional: true},
      
      
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
