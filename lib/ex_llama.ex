defmodule ExLLama do
  use Rustler,
      otp_app: :ex_llama,
      crate: :erlang_llama_cpp_nif

      def load_model(path), do: __nif_load_model__(path, ExLLama.ModelOptions.new())
      def load_model(path, %ExLLama.ModelOptions{} = opts), do: __nif_load_model__(path, opts)
      def __nif_load_model__(_,_), do: :erlang.nif_error(:nif_not_loaded)

end
