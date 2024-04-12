use llama_cpp::Token;
use rustler::NifStruct;

#[derive(NifStruct)]
#[module = "ExLLama.Token"]
pub struct ExLLamaToken {
    pub token: i32,
}

impl From<ExLLamaToken> for Token {
    fn from(value: ExLLamaToken) -> Self {
        Self(value.token)
    }
}


impl From<Token> for ExLLamaToken{
    fn from(value: Token) -> Self {
        Self {
            token: value.0,
        }
    }
}