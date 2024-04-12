use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{SessionParams, Token};
use regex::Regex;
use rustler::ResourceArc;
use crate::refs::session_ref::ExLLamaSessionRef;
use crate::structs::model::ExLLamaModel;
use crate::structs::session::ExLLamaSession;
use crate::structs::session_options::ExLLamaSessionOptions;


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_default_session_options__() -> Result<ExLLamaSessionOptions, String> {
    let r = ExLLamaSessionOptions::from(SessionParams::default());
    Ok(r)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_advance_context_with_tokens__(session:  ResourceArc<ExLLamaSessionRef>, context: Vec<i32>) -> Result<&'static str, String> {
    let mut ctx = session.0.lock().expect("Locking the session failed");
    let tokens: Vec<Token> = context.into_iter().map(Token).collect();
    let result = ctx.advance_context_with_tokens(tokens);
    match result {
        Ok(_) => Ok("OK"),
        Err(e) => Err(e.to_string())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_advance_context__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<&'static str,String> {
    let mut ctx = session.0.lock().expect("Locking the session failed");
    let result = ctx.advance_context(context);
    match result {
        Ok(_) => Ok("OK"),
        Err(e) => Err(e.to_string())
    }
}

// start_completing

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_completion__(session:  ResourceArc<ExLLamaSessionRef>, max_predictions: usize, stop: String) -> Result<String,String> {
    let lock = session.0.lock().expect("Locking the session failed");
    let c = lock.deep_copy()  ;
    match c {
        Ok(ctx) => {
            let mut ctx = ctx;
            let completions = ctx.start_completing_with(StandardSampler::default(), max_predictions).into_strings();
            let mut completions_str = String::new();
            let pattern = Regex::new(&stop).unwrap(); // Compile the regex, handle errors as needed
            // todo real method that returns a reference to completion handler
            for completion in completions {
                completions_str.push_str(&completion);
                if let Some(mat) = pattern.find(&completions_str) {
                    completions_str.truncate(mat.end());
                    break;
                }
            }
            Ok(completions_str)
        },
        Err(e) => Err(e.to_string())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_model__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<ExLLamaModel,String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let model = ctx.model();
    let wrapper = ExLLamaModel::new(model);
    Ok(wrapper)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_params__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<ExLLamaSessionOptions,String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let params = ctx.params();
    let wrapper = ExLLamaSessionOptions::from(params);
    Ok(wrapper)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_context_size__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<usize, String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let result = ctx.context_size();
    Ok(result)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_context__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<Vec<i32>, String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let result = ctx.context();
    let tokens: Vec<i32> = result.into_iter().map(|x| x.0).collect();
    Ok(tokens)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_truncate_context__(session:  ResourceArc<ExLLamaSessionRef>, n_tokens: usize) -> Result<&'static str, String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    ctx.truncate_context(n_tokens);
    Ok("OK")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_set_context_to_tokens__(session:  ResourceArc<ExLLamaSessionRef>, context: Vec<i32>) -> Result<&'static str,String> {
    let mut ctx = session.0.lock().expect("Locking the session failed");
    let tokens: Vec<Token> = context.into_iter().map(Token).collect();
    let result = ctx.set_context_to_tokens(tokens);
    match result {
        Ok(_) => Ok("OK"),
        Err(e) => Err(e.to_string())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_set_context__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<&'static str,String> {
    let mut ctx = session.0.lock().expect("Locking the session failed");
    let result = ctx.set_context(context);
    match result {
        Ok(_) => Ok("OK"),
        Err(e) => Err(e.to_string())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_deep_copy__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<ExLLamaSession, String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let result = ctx.deep_copy();
    match result {
        Ok(session) =>
            Ok(ExLLamaSession::new(session)),
        Err(e) => return Err(e.to_string()),
    }
}
