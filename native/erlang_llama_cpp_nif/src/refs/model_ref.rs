use llama_cpp::LlamaModel;
use rustler::Resource;

pub struct ExLLamaModelRef(pub LlamaModel);

#[rustler::resource_impl]
impl Resource for ExLLamaModelRef {}

impl ExLLamaModelRef {
    pub fn new(llama: LlamaModel) -> Self {
        Self(llama)
    }
}

unsafe impl Send for ExLLamaModelRef {}
unsafe impl Sync for ExLLamaModelRef {}
