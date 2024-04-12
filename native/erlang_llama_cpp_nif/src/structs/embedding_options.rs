use llama_cpp::EmbeddingsParams;
use rustler::NifStruct;

#[derive(NifStruct)]
#[module = "ExLLama.EmbeddingOptions"]
pub struct ExLLamaEmbeddingOptions {
    pub n_threads: u32,
    pub n_threads_batch: u32,
}

impl From<ExLLamaEmbeddingOptions> for EmbeddingsParams {
    fn from(value: ExLLamaEmbeddingOptions) -> Self {
        Self {
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
        }
    }
}


impl From<EmbeddingsParams> for ExLLamaEmbeddingOptions{
    fn from(value: EmbeddingsParams) -> Self {
        Self {
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
        }
    }
}