// This module contains the ExLLama struct which is a wrapper around the LLama struct from the llama_cpp_rs crate.
// The ExLLama struct includes a ResourceArc to the ExLLamaRef struct, which is used to ensure safe concurrent access.
// The ExLLama struct also implements the Deref trait to allow it to be treated as a LLama object.

use std::ops::Deref;
use llama_cpp::LlamaModel;
use rustler::{NifStruct, ResourceArc};
use crate::refs::model_ref::ExLLamaModelRef;


#[derive(NifStruct)]
#[module = "ExLLama.Model"]
// This struct is used to create a resource that can be passed between Elixir and Rust.
pub struct ExLLamaModel {
    pub resource: ResourceArc<ExLLamaModelRef>,
}

// This implementation of ExLLama creates a new instance of the ExLLama struct.
impl ExLLamaModel {
    pub fn new(llama: LlamaModel) -> Self {
        Self {
            resource: ResourceArc::new(ExLLamaModelRef::new(llama)),
        }
    }
}

// This implementation of Deref allows the ExLLama struct to be treated as a LLama object.
impl Deref for ExLLamaModel {
    type Target = LlamaModel;

    fn deref(&self) -> &Self::Target {
        &self.resource.0
    }
}