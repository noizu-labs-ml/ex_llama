// This module contains the Ref Wrapper structs which are a wrapper around the LLama structs from the llama_cpp crate.
// These ref structs are used to ensure safe concurrent access to the LLama objects.
// These ref structs is also marked as Send and Sync, allowing it to be shared across threads.


use llama_cpp::{LlamaModel};


// This struct is used to create a reference to the LLamaModel object.
pub struct ExLLamaModelRef(pub LlamaModel);


impl ExLLamaModelRef {
    pub fn new(llama: LlamaModel) -> Self {
        Self(llama)
    }
}

impl Drop for ExLLamaModelRef {
    fn drop(&mut self) {
        // Log or print a message indicating the resource is being dropped.
        // println!("Dropping ExLLamaModelRef");
    }
}
unsafe impl Send for ExLLamaModelRef {}
unsafe impl Sync for ExLLamaModelRef {}
