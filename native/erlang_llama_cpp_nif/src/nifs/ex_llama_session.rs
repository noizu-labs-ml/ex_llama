use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{SessionParams, Token};
use regex::Regex;
use rustler::{Env, ResourceArc};
use rustler::types::Pid;
use crate::refs::session_ref::ExLLamaSessionRef;
use crate::structs::completion::ExLLamaCompletion;
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
pub fn __session_nif_start_completing_with__(env: Env, pid: Pid, session:  ResourceArc<ExLLamaSessionRef>, max_predictions: usize) -> Result<&'static str,String> {
    let lock = session.0.lock().expect("Locking the session failed");
    let c = lock.deep_copy()  ;
    match c {
        Ok(ctx) => {
            let mut pid = pid;
            let mut ctx = ctx;
            let handle = ctx.start_completing_with(StandardSampler::default(), max_predictions);
            let i = handle.into_strings();
            for completion in i {
                //let gen_completion = rustler::types::tuple::make_tuple(&[rustler::types::atom::from_str("gen"), completion]);
                env.send(&pid, completion).expect("Encoding completion failed");
            }
            let fin = rustler::types::Atom::from_str(env, "fin").expect("Encoding completion failed");
            env.send(&pid, fin).expect("Encoding completion failed");
            Ok("OK")
        },
        Err(e) => Err(e.to_string())
    }
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_completion__(session:  ResourceArc<ExLLamaSessionRef>, max_predictions: usize, stop: Option<String>) -> Result<ExLLamaCompletion,String> {
    let lock = session.0.lock().expect("Locking the session failed");
    let c = lock.deep_copy()  ;
    match c {
        Ok(ctx) => {
            let mut ctx = ctx;
            let prompt_size = ctx.context_size();
            let completions = ctx.start_completing_with(StandardSampler::default(), max_predictions).into_strings();
            let mut completions_str = String::new();

            match stop {
                Some(x) => {
                    let pattern = Regex::new(&x).unwrap(); // Compile the regex, handle errors as needed
                    for completion in completions {
                        completions_str.push_str(&completion);
                        if let Some(mat) = pattern.find(&completions_str) {
                            completions_str.truncate(mat.end());
                            break;
                        }
                    }
                },
                None => {
                    for completion in completions {
                        completions_str.push_str(&completion);
                    }
                }
            }
            Ok(ExLLamaCompletion::new(completions_str, ctx.context_size() - prompt_size))
        },
        Err(e) => Err(e.to_string())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_model__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<ExLLamaModel,String> {
    let ctx = session.0.lock().expect("Locking the session failed");
    let model = ctx.model();
    let wrapper = ExLLamaModel::new("...".to_string(), model);
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
            Ok(ExLLamaSession::new("".to_string(), session.params().seed, session)),
        Err(e) => return Err(e.to_string()),
    }
}
