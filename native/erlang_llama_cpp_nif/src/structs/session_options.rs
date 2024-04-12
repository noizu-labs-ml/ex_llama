use llama_cpp::SessionParams;
use rustler::NifStruct;

#[derive(NifStruct)]
#[module = "ExLLama.SessionOptions"]
pub struct ExLLamaSessionOptions {
    pub seed: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub type_k: u32,
    pub type_v: u32,
    pub embedding: bool,
    pub offload_kqv: bool,
    pub pooling: bool,
}

impl From<ExLLamaSessionOptions> for SessionParams {
    fn from(value: ExLLamaSessionOptions) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            type_k: value.type_k,
            type_v: value.type_v,
            embedding: value.embedding,
            offload_kqv: value.offload_kqv,
            pooling: value.pooling,
        }
    }
}


impl From<SessionParams> for ExLLamaSessionOptions{
    fn from(value: SessionParams) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            type_k: value.type_k,
            type_v: value.type_v,
            embedding: value.embedding,
            offload_kqv: value.offload_kqv,
            pooling: value.pooling,
        }
    }
}


impl From<&SessionParams> for ExLLamaSessionOptions{
    fn from(value: &SessionParams) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            type_k: value.type_k,
            type_v: value.type_v,
            embedding: value.embedding,
            offload_kqv: value.offload_kqv,
            pooling: value.pooling,
        }
    }
}