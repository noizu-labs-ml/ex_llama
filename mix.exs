defmodule ExLLama.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_llama,
      name: "LLama CPP Nif Wrapper",
      description: description(),
      package: package(),
      version: "0.0.1",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
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

  defp package() do
    [
      licenses: ["MIT"],
      links: %{
        project: "https://github.com/noizu-labs-ml/ex_llama",
        developer_github: "https://github.com/noizu"
      }
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

      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
