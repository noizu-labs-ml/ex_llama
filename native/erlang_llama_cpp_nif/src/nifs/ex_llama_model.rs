use llama_cpp::{EmbeddingsParams, LlamaModel, LlamaParams, SessionParams, Token};
use crate::structs::embedding_options::ExLLamaEmbeddingOptions;
use crate::structs::model::ExLLamaModel;
use crate::structs::model_options::ModelOptions;
use crate::structs::session::ExLLamaSession;
use crate::structs::session_options::ExLLamaSessionOptions;


#[rustler::nif(schedule = "DirtyCpu")]
// This function creates a new instance of the ExLLama struct.
pub fn __model_nif_load_from_file__(path: String, model_options: ModelOptions) -> Result<ExLLamaModel, String> {
    let params = LlamaParams::from(model_options);
    let model = LlamaModel::load_from_file(path, params);
    match model {
        Ok(model) =>
            Ok(ExLLamaModel::new(model)),
        Err(e) =>
            Err(e.to_string()),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_detokenize__(model: ExLLamaModel, token: i32) -> Result<Vec<u8>, String> {
    let t = Token(token);
    let x = model.detokenize(t);
    let vector = Vec::from(x);
    Ok(vector)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_token_to_byte_piece__(model: ExLLamaModel, token: i32) -> Result<Vec<u8>, String> {
    let t = Token(token);
    let x = model.token_to_byte_piece(t);
    Ok(x)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_token_to_piece__(model: ExLLamaModel, token: i32) -> Result<String, String> {
    let t = Token(token);
    let x = model.token_to_piece(t);
    Ok(x)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_decode_tokens__(model: ExLLamaModel, tokens: Vec<i32>) -> Result<String, String> {
    let tokens: Vec<Token> = tokens.into_iter().map(Token).collect();
    let x = model.decode_tokens(tokens);
    Ok(x)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_create_session__(model: ExLLamaModel, options: ExLLamaSessionOptions) -> Result<ExLLamaSession, String> {
    let opts = SessionParams::from(options);
    let ctx = model.create_session(opts);
    match ctx {
        Ok(session) =>
            Ok(ExLLamaSession::new(session)),
        Err(e) => return Err(e.to_string()),
    }
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_embeddings__(model: ExLLamaModel, inputs: Vec<u8>, options: ExLLamaEmbeddingOptions) -> Result<Vec<Vec<f32>>, String> {
    let options = EmbeddingsParams::from(options);
    let vec_of_vec = vec![inputs];
    let response = model.embeddings(&vec_of_vec, options);
    match response {
        Ok(value) =>
            Ok(value),
        Err(e) => return Err(e.to_string()),
    }
}

// @TODO embeddings_async

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_bos__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.bos();
    Ok(x.0)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_eos__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.eos();
    Ok(x.0)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_nl__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.nl();
    Ok(x.0)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_prefix__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.infill_prefix();
    Ok(x.0)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_middle__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.infill_middle();
    Ok(x.0)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_infill_suffix__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.infill_suffix();
    Ok(x.0)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_eot__(model: ExLLamaModel) -> Result<i32, ()> {
    let x = model.eot();
    Ok(x.0)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_vocabulary_size__(model: ExLLamaModel) -> Result<usize, ()> {
    let x = model.vocabulary_size();
    Ok(x)
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_embed_len__(model: ExLLamaModel) -> Result<usize, ()> {
    let x = model.embed_len();
    Ok(x)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __model_nif_train_len__(model: ExLLamaModel) -> Result<usize, ()> {
    let x = model.train_len();
    Ok(x)
}
