use std::sync::Mutex;
use llama_cpp::LlamaSession;
use rustler::Resource;

pub struct ExLLamaSessionRef(pub Mutex<LlamaSession>);

#[rustler::resource_impl]
impl Resource for ExLLamaSessionRef {}

impl ExLLamaSessionRef {
    pub fn new(session: LlamaSession) -> Self {
        Self(Mutex::new(session))
    }
}

unsafe impl Send for ExLLamaSessionRef {}
unsafe impl Sync for ExLLamaSessionRef {}
