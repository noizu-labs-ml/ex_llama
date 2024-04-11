mod nifs;
mod structs;
mod refs;
use crate::nifs::__nif_load_model__;
use rustler::{Env, Term};
use crate::refs::model_ref::ExLLamaModelRef;

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExLLamaModelRef, env);

    true
}

rustler::init!(
    "Elixir.ExLLama",
    [__nif_load_model__],
    load = on_load
);
