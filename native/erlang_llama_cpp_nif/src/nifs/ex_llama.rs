
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use llama_cpp::standard_sampler::StandardSampler;
// use rustler::ResourceArc;
// use crate::refs::model_ref::ExLLamaModelRef;
use crate::structs::model::ExLLamaModel;
use crate::structs::model_options::ModelOptions;
use crate::structs::session::ExLLamaSession;
use crate::structs::session_options::ExLLamaSessionOptions;
extern crate regex;
use regex::Regex;
use rustler::ResourceArc;
use crate::refs::model_ref::ExLLamaModelRef;
use crate::refs::session_ref::ExLLamaSessionRef;

#[rustler::nif(schedule = "DirtyCpu")]
// This function creates a new instance of the ExLLama struct.
pub fn __context_nif_load_model__(path: String, model_options: ModelOptions) -> Result<ExLLamaModel, ()> {
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
pub fn __context_nif_default_session_options__() -> ExLLamaSessionOptions {
    ExLLamaSessionOptions::from(SessionParams::default())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __context_nif_create_session__(model: ExLLamaModel, options: ExLLamaSessionOptions) -> Result<ExLLamaSession, ()> {
    let opts = SessionParams::from(options);
    let ctx = model.create_session(opts);
    match ctx {
        Ok(session) =>
            Ok(ExLLamaSession::new(session)),
        Err(_) => return Err(()),
    }
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __context_nif_advance_context__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<(), &'static str> {
    //let mut ctx = session.lock().map_err(|_| "Failed to acquire lock")?;
    let mut ctx = session.0.lock().expect("Locking the session failed");

    ctx.advance_context(context).map_err(|_| "Error advancing context")?;
    Ok(())
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __context_nif_complete__(session: ResourceArc<ExLLamaSessionRef>, tokens: usize) -> Result<String, &'static str> {
    let mut ctx2 = session.0.lock().expect("Locking the session failed");
    let mut ctx = ctx2.clone();
    let mut completions = ctx.start_completing_with(StandardSampler::default(), tokens).into_strings();
    let mut completions_str = String::new();
    let pattern = Regex::new(r"</s>").unwrap(); // Compile the regex, handle errors as needed

    for completion in completions {
        completions_str.push_str(&completion);
        if let Some(mat) = pattern.find(&completions_str) {
            completions_str.truncate(mat.start());
            completions_str.push_str("</s>");
            break;
        }
    }

    Ok(completions_str)
}
