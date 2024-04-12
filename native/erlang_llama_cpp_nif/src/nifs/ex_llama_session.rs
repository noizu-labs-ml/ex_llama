use rustler::ResourceArc;
use crate::refs::session_ref::ExLLamaSessionRef;
use crate::structs::session::ExLLamaSession;

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_advance_context_with_tokens__(session:  ResourceArc<ExLLamaSessionRef>, context: Vec<i32>) -> Result<(), &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_advance_context__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<(), &'static str> {
    Err("error")
}

// start_completing

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_completion__(session:  ResourceArc<ExLLamaSessionRef>, max_predictions: u32) -> Result<String, &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_model__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<String, &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_params__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<String, &'static str> {
    Err("error")
}


#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_context_size__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<usize, &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_context__(session:  ResourceArc<ExLLamaSessionRef>) -> Result<Vec<i32>, &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_truncate_context__(session:  ResourceArc<ExLLamaSessionRef>, n_tokens: usize) -> Result<Vec<i32>, &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_set_context_to_tokens__(session:  ResourceArc<ExLLamaSessionRef>, tokens: Vec<i32>) -> Result<(), &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_nif_set_context__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<(), &'static str> {
    Err("error")
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn __session_deep_copy__(session:  ResourceArc<ExLLamaSessionRef>, context: String) -> Result<ExLLamaSession, &'static str> {
    Err("error")
}