// This module contains the ExLLama struct which is a wrapper around the LLama struct from the llama_cpp_rs crate.
// The ExLLama struct includes a ResourceArc to the ExLLamaRef struct, which is used to ensure safe concurrent access.
// The ExLLama struct also implements the Deref trait to allow it to be treated as a LLama object.


use llama_cpp::{LlamaSession};
use rustler::{NifStruct, ResourceArc};
use crate::refs::session_ref::ExLLamaSessionRef;


#[derive(NifStruct)]
#[module = "ExLLama.Session"]
pub struct ExLLamaSession {
    pub resource: ResourceArc<ExLLamaSessionRef>,
}

impl ExLLamaSession {
    pub fn new(session: LlamaSession) -> Self {
        Self {
            resource: ResourceArc::new(ExLLamaSessionRef::new(session)),
        }
    }
    //
    // // Provide a method to access the mutex protected session
    // pub fn lock_session(&self) -> std::sync::MutexGuard<'_, LlamaSession> {
    //     self.resource.0.lock().expect("Locking the session failed")
    // }
}