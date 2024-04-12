// This module contains the ExLLama struct which is a wrapper around the LLama struct from the llama_cpp_rs crate.
// The ExLLama struct includes a ResourceArc to the ExLLamaRef struct, which is used to ensure safe concurrent access.
// The ExLLama struct also implements the Deref trait to allow it to be treated as a LLama object.


use llama_cpp::{LlamaSession};
use rustler::{NifStruct, ResourceArc};
use crate::refs::session_ref::ExLLamaSessionRef;


#[derive(NifStruct)]
#[module = "ExLLama.Completion"]
pub struct ExLLamaCompletion {
    pub content: String,
    pub token_length: usize,
}

impl ExLLamaCompletion {
    pub fn new(content: String, token_length: usize) -> Self {
        Self {
            content: content,
            token_length: token_length
        }
    }
    //
    // // Provide a method to access the mutex protected session
    // pub fn lock_session(&self) -> std::sync::MutexGuard<'_, LlamaSession> {
    //     self.resource.0.lock().expect("Locking the session failed")
    // }
}