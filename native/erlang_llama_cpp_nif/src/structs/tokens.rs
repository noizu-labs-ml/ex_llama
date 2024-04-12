use llama_cpp::Token;
use rustler::NifStruct;

#[derive(NifStruct)]
#[module = "ExLLama.Tokens"]
pub struct ExLLamaTokens {
    pub token: Vec<i32>,
}