use llama_cpp::{LlamaParams, SplitMode};
use rustler::{NifStruct};
//use rustler::types::atom;

#[derive(NifStruct)]
#[module = "ExLLama.ModelOptions"]
pub struct ModelOptions {
    pub n_gpu_layers: u32,
    pub split_mode: String,
    pub main_gpu: u32,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl From<ModelOptions> for LlamaParams {
    fn from(value: ModelOptions) -> Self {
        // Map the string values to the corresponding SplitMode enum values
        let split_mode = match value.split_mode.as_str() {
            "none" => SplitMode::None,
            "layer" => SplitMode::Layer,
            "row" => SplitMode::Row,
            _ => panic!("Invalid split_mode value"),
        };

        Self {
            n_gpu_layers: value.n_gpu_layers,
            split_mode: split_mode,
            main_gpu: value.main_gpu,
            vocab_only: value.vocab_only,
            use_mmap: value.use_mmap,
            use_mlock: value.use_mlock,
        }
    }
}