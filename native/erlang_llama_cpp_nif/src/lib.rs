mod nifs;
mod structs;
mod refs;
use crate::nifs::{__nif_default_session_options__, __nif_load_model__,__nif_create_session__,__nif_advance_context__,__nif_complete__};
use rustler::{Env, Term};
use crate::refs::model_ref::ExLLamaModelRef;
use crate::refs::session_ref::ExLLamaSessionRef;
use crate::structs::session::ExLLamaSession;

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExLLamaModelRef, env);
    rustler::resource!(ExLLamaSessionRef, env);
    rustler::resource!(ExLLamaSession, env);

    true
}

rustler::init!(
    "Elixir.ExLLama",
    [__nif_load_model__,__nif_default_session_options__,__nif_create_session__,__nif_advance_context__,__nif_complete__],
    load = on_load
);
