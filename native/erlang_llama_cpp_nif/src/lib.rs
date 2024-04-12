mod nifs;
mod structs;
mod refs;
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
    "Elixir.ExLLama.Nif",
    [
        nifs::ex_llama_model::__model_nif_load_from_file__,
        nifs::ex_llama_model::__model_nif_detokenize__,
        nifs::ex_llama_model::__model_nif_token_to_byte_piece__,
        nifs::ex_llama_model::__model_nif_token_to_piece__,
        nifs::ex_llama_model::__model_nif_decode_tokens__,
        nifs::ex_llama_model::__model_nif_create_session__,
        nifs::ex_llama_model::__model_nif_embeddings__,
        nifs::ex_llama_model::__model_nif_bos__,
        nifs::ex_llama_model::__model_nif_eos__,
        nifs::ex_llama_model::__model_nif_nl__,
        nifs::ex_llama_model::__model_nif_infill_prefix__,
        nifs::ex_llama_model::__model_nif_infill_middle__,
        nifs::ex_llama_model::__model_nif_infill_suffix__,
        nifs::ex_llama_model::__model_nif_eot__,
        nifs::ex_llama_model::__model_nif_vocabulary_size__,
        nifs::ex_llama_model::__model_nif_embed_len__,
        nifs::ex_llama_model::__model_nif_train_len__,

        nifs::ex_llama_session::__session_nif_default_session_options__,
        nifs::ex_llama_session::__session_nif_advance_context_with_tokens__,
        nifs::ex_llama_session::__session_nif_advance_context__,
        nifs::ex_llama_session::__session_nif_completion__,
        nifs::ex_llama_session::__session_nif_model__,
        nifs::ex_llama_session::__session_nif_params__,
        nifs::ex_llama_session::__session_nif_context_size__,
        nifs::ex_llama_session::__session_nif_context__,
        nifs::ex_llama_session::__session_nif_truncate_context__,
        nifs::ex_llama_session::__session_nif_set_context_to_tokens__,
        nifs::ex_llama_session::__session_nif_set_context__,
        nifs::ex_llama_session::__session_deep_copy__,
        //
        // nifs::ex_llama::__context_nif_load_model__,
        // nifs::ex_llama::__context_nif_default_session_options__,
        // nifs::ex_llama::__context_nif_create_session__,
        // nifs::ex_llama::__context_nif_advance_context__,
        // nifs::ex_llama::__context_nif_complete__,

    ],
    load = on_load
);
