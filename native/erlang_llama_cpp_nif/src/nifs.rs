
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use rustler::ResourceArc;
use crate::refs::model_ref::ExLLamaModelRef;
use crate::structs::model::ExLLamaModel;
use crate::structs::model_options::ModelOptions;

#[rustler::nif(schedule = "DirtyCpu")]
// This function creates a new instance of the ExLLama struct.
pub fn __nif_load_model__(path: String, model_options: ModelOptions) -> Result<ExLLamaModel, ()> {
    let params = LlamaParams::from(model_options);
    let model = LlamaModel::load_from_file(path, params);
    match model {
        Ok(model) =>
            Ok(ExLLamaModel::new(model)),
        Err(_) =>
            Err(()),
    }
}
