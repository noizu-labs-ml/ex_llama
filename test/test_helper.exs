#  Download Required  Models
# TODO add streaming download manager with download status.
unless File.exists?("test/models/phi-2-orange.Q2_K.gguf") do
  IO.puts "Downloading required model: phi-2-orange"
  IO.puts "navigate to test/models and run `wget -O phi-2-orange.Q2_K.gguf https://huggingface.co/TheBloke/phi-2-orange-GGUF/resolve/main/phi-2-orange.Q2_K.gguf&download=true`"
  IO.puts "and `wget -O tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true`"
  File.mkdir_p("test/models")
end

ExUnit.start()
