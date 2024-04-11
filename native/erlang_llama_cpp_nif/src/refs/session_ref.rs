// This module contains the Ref Wrapper structs which are a wrapper around the LLama structs from the llama_cpp crate.
// These ref structs are used to ensure safe concurrent access to the LLama objects.
// These ref structs is also marked as Send and Sync, allowing it to be shared across threads.


use std::sync::Mutex;
use llama_cpp::{LlamaSession};

pub struct ExLLamaSessionRef(pub Mutex<LlamaSession>);

impl ExLLamaSessionRef {
    pub fn new(session: LlamaSession) -> Self {
        Self(Mutex::new(session))
    }
}

impl Drop for ExLLamaSessionRef {
    fn drop(&mut self) {
        println!("Dropping ExLLamaSessionRef");
    }
}

unsafe impl Send for ExLLamaSessionRef {}
unsafe impl Sync for ExLLamaSessionRef {}