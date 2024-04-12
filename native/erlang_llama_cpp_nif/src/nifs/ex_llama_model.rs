use llama_cpp::{LlamaModel, LlamaParams, SessionParams, Token};
use crate::structs::embedding_options::ExLLamaEmbeddingOptions;
use crate::structs::model::ExLLamaModel;
use crate::structs::model_options::ModelOptions;
use crate::structs::session::ExLLamaSession;
use crate::structs::session_options::ExLLamaSessionOptions;
use crate::structs::token::ExLLamaToken;
use crate::structs::tokens::ExLLamaTokens;


#[rustler::nif(schedule = "DirtyCpu")]
// This function creates a new instance of the ExLLama struct.
pub fn __model_nif_load_from_file__(path: String, model_options: ModelOptions) -> Result<ExLLamaModel, ()> {
    let params = LlamaParams::from(model_options);
    let model = LlamaModel::load_from_file(path, params);
    match model {
        Ok(model) =>
            Ok(ExLLamaModel::new(model)),
        Err(_) =>
            Err(()),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_detokenize__(model: ExLLamaModel, token: i32) -> Result<Vec<u8>, ()> {
    return Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_token_to_byte_piece__(model: ExLLamaModel, token: i32) -> Result<Vec<u8>, ()> {
    return Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_token_to_piece__(model: ExLLamaModel, token: i32) -> Result<String, ()> {
    return Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_decode_tokens__(model: ExLLamaModel, tokens: Vec<i32>) -> Result<String, ()> {
    return Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_create_session__(model: ExLLamaModel, options: ExLLamaSessionOptions) -> Result<ExLLamaSession, ()> {
    let opts = SessionParams::from(options);
    let ctx = model.create_session(opts);
    match ctx {
        Ok(session) =>
            Ok(ExLLamaSession::new(session)),
        Err(_) => return Err(()),
    }
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_embeddings__(model: ExLLamaModel, inputs: Vec<u8>, options: ExLLamaEmbeddingOptions) -> Result<Vec<Vec<f32>>, ()> {
    Err(())
}

// @TODO embeddings_async

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_bos__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_eos__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_nl__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_prefix__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_middle__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_suffix__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_eot__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_vocabulary_size__(model: ExLLamaModel) -> Result<i32, ()> {
    Err(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_embed_len__(model: ExLLamaModel) -> Result<usize, ()> {
    Err(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_train_len__(model: ExLLamaModel) -> Result<usize, ()> {
    Err(())
}
