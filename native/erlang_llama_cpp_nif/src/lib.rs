#![allow(non_local_definitions)]
mod nifs;
mod structs;
mod refs;

rustler::init!("Elixir.ExLLama.Nif");
