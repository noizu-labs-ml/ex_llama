#include <erl_nif.h>
#include <llama.h>
#include <ggml.h>

#include <string>
#include <vector>
#include <cstring>
#include <mutex>
#include <thread>
#include <cmath>
#include <regex>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Resource types
// ---------------------------------------------------------------------------
static ErlNifResourceType *MODEL_RES_TYPE = nullptr;
static ErlNifResourceType *CTX_RES_TYPE   = nullptr;

struct ModelRes {
    llama_model *model;
};

struct CtxRes {
    llama_context *ctx;
    llama_model   *model;   // non-owning, for vocab access
    std::mutex     mu;
};

static void model_res_dtor(ErlNifEnv *, void *obj) {
    auto *r = static_cast<ModelRes *>(obj);
    if (r->model) llama_model_free(r->model);
}

static void ctx_res_dtor(ErlNifEnv *, void *obj) {
    auto *r = static_cast<CtxRes *>(obj);
    if (r->ctx) llama_free(r->ctx);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static ERL_NIF_TERM make_atom(ErlNifEnv *env, const char *name) {
    ERL_NIF_TERM atom;
    if (enif_make_existing_atom(env, name, &atom, ERL_NIF_LATIN1))
        return atom;
    return enif_make_atom(env, name);
}

static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM val) {
    return enif_make_tuple2(env, make_atom(env, "ok"), val);
}

static ERL_NIF_TERM make_error(ErlNifEnv *env, const char *reason) {
    return enif_make_tuple2(env, make_atom(env, "error"),
                            enif_make_string(env, reason, ERL_NIF_LATIN1));
}

static ERL_NIF_TERM make_error_bin(ErlNifEnv *env, const std::string &reason) {
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, reason.size(), &bin);
    memcpy(buf, reason.data(), reason.size());
    return enif_make_tuple2(env, make_atom(env, "error"), bin);
}

static bool get_string(ErlNifEnv *env, ERL_NIF_TERM term, std::string &out) {
    ErlNifBinary bin;
    if (enif_inspect_binary(env, term, &bin)) {
        out.assign(reinterpret_cast<const char *>(bin.data), bin.size);
        return true;
    }
    if (enif_inspect_iolist_as_binary(env, term, &bin)) {
        out.assign(reinterpret_cast<const char *>(bin.data), bin.size);
        return true;
    }
    return false;
}

static bool get_uint(ErlNifEnv *env, ERL_NIF_TERM term, unsigned int &out) {
    return enif_get_uint(env, term, &out);
}

static bool get_int(ErlNifEnv *env, ERL_NIF_TERM term, int &out) {
    return enif_get_int(env, term, &out);
}

static bool get_bool(ErlNifEnv *env, ERL_NIF_TERM term, bool &out) {
    char buf[8];
    if (!enif_get_atom(env, term, buf, sizeof(buf), ERL_NIF_LATIN1))
        return false;
    if (strcmp(buf, "true") == 0) { out = true; return true; }
    if (strcmp(buf, "false") == 0) { out = false; return true; }
    return false;
}

static bool map_get_uint(ErlNifEnv *env, ERL_NIF_TERM map,
                         const char *key, unsigned int &out) {
    ERL_NIF_TERM val;
    if (enif_get_map_value(env, map, make_atom(env, key), &val))
        return get_uint(env, val, out);
    return false;
}

static bool map_get_int(ErlNifEnv *env, ERL_NIF_TERM map,
                        const char *key, int &out) {
    ERL_NIF_TERM val;
    if (enif_get_map_value(env, map, make_atom(env, key), &val))
        return get_int(env, val, out);
    return false;
}

static bool map_get_bool(ErlNifEnv *env, ERL_NIF_TERM map,
                         const char *key, bool &out) {
    ERL_NIF_TERM val;
    if (enif_get_map_value(env, map, make_atom(env, key), &val))
        return get_bool(env, val, out);
    return false;
}

static bool map_get_double(ErlNifEnv *env, ERL_NIF_TERM map,
                           const char *key, double &out) {
    ERL_NIF_TERM val;
    if (enif_get_map_value(env, map, make_atom(env, key), &val))
        return enif_get_double(env, val, &out);
    return false;
}

static bool map_get_string(ErlNifEnv *env, ERL_NIF_TERM map,
                           const char *key, std::string &out) {
    ERL_NIF_TERM val;
    if (enif_get_map_value(env, map, make_atom(env, key), &val))
        return get_string(env, val, out);
    return false;
}

static ERL_NIF_TERM make_binary_string(ErlNifEnv *env, const std::string &s) {
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, s.size(), &bin);
    memcpy(buf, s.data(), s.size());
    return bin;
}

// Tokenize a string using vocab
static std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                          const std::string &text,
                                          bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0,
                           add_special, true);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens(n);
    int actual = llama_tokenize(vocab, text.c_str(), text.size(),
                                tokens.data(), tokens.size(),
                                add_special, true);
    if (actual < 0) actual = -actual;
    tokens.resize(actual);
    return tokens;
}

// Detokenize a single token
static std::string token_to_piece(const llama_vocab *vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::vector<char> big(-n);
        int n2 = llama_token_to_piece(vocab, tok, big.data(), big.size(), 0, true);
        if (n2 < 0) return "";
        return std::string(big.data(), n2);
    }
    return std::string(buf, n);
}

// Detokenize a token vector
static std::string detokenize(const llama_vocab *vocab,
                               const std::vector<llama_token> &tokens) {
    std::string out;
    for (auto t : tokens) out += token_to_piece(vocab, t);
    return out;
}

// ---------------------------------------------------------------------------
// NIF: load_model(path, opts_map) -> {:ok, model_ref} | {:error, reason}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_load_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    std::string path;
    if (!get_string(env, argv[0], path))
        return make_error(env, "bad path argument");

    llama_model_params params = llama_model_default_params();

    if (argc > 1 && enif_is_map(env, argv[1])) {
        unsigned int u;
        bool b;
        if (map_get_uint(env, argv[1], "n_gpu_layers", u))
            params.n_gpu_layers = (int32_t)u;
        if (map_get_bool(env, argv[1], "vocab_only", b))
            params.vocab_only = b;
        if (map_get_bool(env, argv[1], "use_mmap", b))
            params.use_mmap = b;
        if (map_get_bool(env, argv[1], "use_mlock", b))
            params.use_mlock = b;
    }

    llama_model *model = nullptr;
    try {
        model = llama_model_load_from_file(path.c_str(), params);
    } catch (const std::exception &e) {
        return make_error_bin(env, std::string("model load exception: ") + e.what());
    } catch (...) {
        return make_error(env, "model load: unknown exception");
    }

    if (!model)
        return make_error(env, "failed to load model");

    auto *res = static_cast<ModelRes *>(
        enif_alloc_resource(MODEL_RES_TYPE, sizeof(ModelRes)));
    res->model = model;

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return make_ok(env, term);
}

// ---------------------------------------------------------------------------
// NIF: create_context(model_ref, opts_map) -> {:ok, ctx_ref} | {:error, ...}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_create_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    llama_context_params params = llama_context_default_params();

    if (argc > 1 && enif_is_map(env, argv[1])) {
        unsigned int u;
        int i;
        double d;
        bool b;
        if (map_get_uint(env, argv[1], "n_ctx", u))       params.n_ctx = u;
        if (map_get_uint(env, argv[1], "n_batch", u))     params.n_batch = u;
        if (map_get_int(env, argv[1], "n_threads", i))    params.n_threads = i;
        if (map_get_int(env, argv[1], "n_threads_batch", i)) params.n_threads_batch = i;
        if (map_get_double(env, argv[1], "rope_freq_base", d))  params.rope_freq_base = (float)d;
        if (map_get_double(env, argv[1], "rope_freq_scale", d)) params.rope_freq_scale = (float)d;
        if (map_get_bool(env, argv[1], "embeddings", b))  params.embeddings = b;
        if (map_get_bool(env, argv[1], "offload_kqv", b)) params.offload_kqv = b;
    }

    llama_context *ctx = llama_init_from_model(mres->model, params);
    if (!ctx)
        return make_error(env, "failed to create context");

    auto *cres = static_cast<CtxRes *>(
        enif_alloc_resource(CTX_RES_TYPE, sizeof(CtxRes)));
    new (cres) CtxRes();
    cres->ctx   = ctx;
    cres->model = mres->model;

    ERL_NIF_TERM term = enif_make_resource(env, cres);
    enif_release_resource(cres);
    return make_ok(env, term);
}

// ---------------------------------------------------------------------------
// NIF: tokenize(model_ref, text, add_special) -> {:ok, [token_ids]}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_tokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    std::string text;
    if (!get_string(env, argv[1], text))
        return make_error(env, "bad text argument");

    bool add_special = true;
    if (argc > 2) get_bool(env, argv[2], add_special);

    const llama_vocab *vocab = llama_model_get_vocab(mres->model);
    auto tokens = tokenize(vocab, text, add_special);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);

    return make_ok(env, list);
}

// ---------------------------------------------------------------------------
// NIF: detokenize(model_ref, [token_ids]) -> {:ok, binary}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_detokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    unsigned int len;
    if (!enif_get_list_length(env, argv[1], &len))
        return make_error(env, "bad token list");

    std::vector<llama_token> tokens(len);
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < len; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        int tok;
        enif_get_int(env, head, &tok);
        tokens[i] = tok;
    }

    const llama_vocab *vocab = llama_model_get_vocab(mres->model);
    std::string text = detokenize(vocab, tokens);
    return make_ok(env, make_binary_string(env, text));
}

// ---------------------------------------------------------------------------
// NIF: token_to_piece(model_ref, token_id) -> {:ok, binary}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_token_to_piece(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    int tok;
    if (!enif_get_int(env, argv[1], &tok))
        return make_error(env, "bad token id");

    const llama_vocab *vocab = llama_model_get_vocab(mres->model);
    std::string piece = token_to_piece(vocab, tok);
    return make_ok(env, make_binary_string(env, piece));
}

// ---------------------------------------------------------------------------
// NIF: special tokens - bos, eos, eot, nl, fim_pre, fim_mid, fim_suf
// ---------------------------------------------------------------------------
#define VOCAB_TOKEN_NIF(name, fn) \
static ERL_NIF_TERM \
name(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) \
{ \
    ModelRes *mres; \
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres)) \
        return make_error(env, "bad model reference"); \
    const llama_vocab *vocab = llama_model_get_vocab(mres->model); \
    return make_ok(env, enif_make_int(env, fn(vocab))); \
}

VOCAB_TOKEN_NIF(nif_vocab_bos,     llama_vocab_bos)
VOCAB_TOKEN_NIF(nif_vocab_eos,     llama_vocab_eos)
VOCAB_TOKEN_NIF(nif_vocab_eot,     llama_vocab_eot)
VOCAB_TOKEN_NIF(nif_vocab_nl,      llama_vocab_nl)
VOCAB_TOKEN_NIF(nif_vocab_fim_pre, llama_vocab_fim_pre)
VOCAB_TOKEN_NIF(nif_vocab_fim_mid, llama_vocab_fim_mid)
VOCAB_TOKEN_NIF(nif_vocab_fim_suf, llama_vocab_fim_suf)

// ---------------------------------------------------------------------------
// NIF: model_info(model_ref) -> {:ok, map}
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_model_info(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    const llama_vocab *vocab = llama_model_get_vocab(mres->model);

    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "n_vocab"),
                      enif_make_int(env, llama_vocab_n_tokens(vocab)), &map);
    enif_make_map_put(env, map, make_atom(env, "n_embd"),
                      enif_make_int(env, llama_model_n_embd(mres->model)), &map);
    enif_make_map_put(env, map, make_atom(env, "n_ctx_train"),
                      enif_make_int(env, llama_model_n_ctx_train(mres->model)), &map);
    enif_make_map_put(env, map, make_atom(env, "n_layer"),
                      enif_make_int(env, llama_model_n_layer(mres->model)), &map);
    enif_make_map_put(env, map, make_atom(env, "model_size"),
                      enif_make_uint64(env, llama_model_size(mres->model)), &map);
    enif_make_map_put(env, map, make_atom(env, "n_params"),
                      enif_make_uint64(env, llama_model_n_params(mres->model)), &map);

    return make_ok(env, map);
}

// ---------------------------------------------------------------------------
// NIF: completion(ctx_ref, prompt, opts_map) -> {:ok, map} | {:error, ...}
//   opts: %{max_tokens, seed, temp, top_p, top_k, min_p, stop}
//   stop is a regex string for early stopping
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_completion(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    CtxRes *cres;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&cres))
        return make_error(env, "bad context reference");

    std::string prompt;
    if (!get_string(env, argv[1], prompt))
        return make_error(env, "bad prompt argument");

    // Defaults
    int max_tokens = 512;
    unsigned int seed = LLAMA_DEFAULT_SEED;
    float temp = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    float min_p = 0.05f;
    std::string stop_pattern;

    if (argc > 2 && enif_is_map(env, argv[2])) {
        int i; unsigned int u; double d;
        if (map_get_int(env, argv[2], "max_tokens", i))  max_tokens = i;
        if (map_get_uint(env, argv[2], "seed", u))       seed = u;
        if (map_get_double(env, argv[2], "temp", d))     temp = (float)d;
        if (map_get_double(env, argv[2], "top_p", d))    top_p = (float)d;
        if (map_get_int(env, argv[2], "top_k", i))       top_k = i;
        if (map_get_double(env, argv[2], "min_p", d))    min_p = (float)d;
        map_get_string(env, argv[2], "stop", stop_pattern);
    }

    std::lock_guard<std::mutex> lock(cres->mu);

    const llama_vocab *vocab = llama_model_get_vocab(cres->model);

    auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty())
        return make_error(env, "tokenization produced no tokens");

    uint32_t n_ctx = llama_n_ctx(cres->ctx);
    if (tokens.size() + max_tokens > n_ctx)
        return make_error(env, "prompt + max_tokens exceeds context size");

    llama_memory_clear(llama_get_memory(cres->ctx), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(cres->ctx, batch) != 0)
        return make_error(env, "failed to decode prompt");

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

    // Compile stop regex if provided
    std::regex stop_regex;
    bool has_stop = !stop_pattern.empty();
    if (has_stop) {
        try {
            stop_regex = std::regex(stop_pattern);
        } catch (const std::regex_error &) {
            llama_sampler_free(sampler);
            return make_error(env, "invalid stop regex pattern");
        }
    }

    std::string result;
    int n_generated = 0;
    bool stopped_by_regex = false;

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler, cres->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token))
            break;

        result += token_to_piece(vocab, new_token);
        n_generated++;

        // Check stop pattern
        if (has_stop) {
            std::smatch match;
            if (std::regex_search(result, match, stop_regex)) {
                result = result.substr(0, match.position() + match.length());
                stopped_by_regex = true;
                break;
            }
        }

        llama_batch next = llama_batch_get_one(&new_token, 1);
        if (llama_decode(cres->ctx, next) != 0)
            break;
    }

    llama_sampler_free(sampler);

    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "content"),
                      make_binary_string(env, result), &map);
    enif_make_map_put(env, map, make_atom(env, "token_length"),
                      enif_make_int(env, n_generated), &map);

    return make_ok(env, map);
}

// ---------------------------------------------------------------------------
// NIF: streaming_completion(ctx_ref, prompt, pid, opts_map)
//   Sends each token as binary to pid, then :fin
//   On error sends "error: <detail>" string matching old Rust behavior
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_streaming_completion(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    CtxRes *cres;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&cres))
        return make_error(env, "bad context reference");

    std::string prompt;
    if (!get_string(env, argv[1], prompt))
        return make_error(env, "bad prompt argument");

    ErlNifPid pid;
    if (!enif_get_local_pid(env, argv[2], &pid))
        return make_error(env, "bad pid argument");

    int max_tokens = 512;
    unsigned int seed = LLAMA_DEFAULT_SEED;
    float temp = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    float min_p = 0.05f;

    if (argc > 3 && enif_is_map(env, argv[3])) {
        int i; unsigned int u; double d;
        if (map_get_int(env, argv[3], "max_tokens", i))  max_tokens = i;
        if (map_get_uint(env, argv[3], "seed", u))       seed = u;
        if (map_get_double(env, argv[3], "temp", d))     temp = (float)d;
        if (map_get_double(env, argv[3], "top_p", d))    top_p = (float)d;
        if (map_get_int(env, argv[3], "top_k", i))       top_k = i;
        if (map_get_double(env, argv[3], "min_p", d))    min_p = (float)d;
    }

    enif_keep_resource(cres);

    std::thread([cres, prompt, pid, max_tokens, seed, temp, top_p, top_k, min_p]() {
        std::lock_guard<std::mutex> lock(cres->mu);

        ErlNifEnv *msg_env = enif_alloc_env();
        const llama_vocab *vocab = llama_model_get_vocab(cres->model);

        auto send_error = [&](const std::string &detail) {
            enif_clear_env(msg_env);
            std::string err_msg = "error: " + detail;
            ERL_NIF_TERM bin;
            unsigned char *buf = enif_make_new_binary(msg_env, err_msg.size(), &bin);
            memcpy(buf, err_msg.data(), err_msg.size());
            enif_send(nullptr, &pid, msg_env, bin);
        };

        auto tokens = tokenize(vocab, prompt, true);

        llama_memory_clear(llama_get_memory(cres->ctx), true);
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

        if (llama_decode(cres->ctx, batch) != 0) {
            send_error("failed to decode prompt");
            enif_free_env(msg_env);
            enif_release_resource(cres);
            return;
        }

        auto sparams = llama_sampler_chain_default_params();
        llama_sampler *sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

        for (int i = 0; i < max_tokens; i++) {
            llama_token new_token = llama_sampler_sample(sampler, cres->ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token))
                break;

            std::string piece = token_to_piece(vocab, new_token);

            enif_clear_env(msg_env);
            ERL_NIF_TERM bin;
            unsigned char *buf = enif_make_new_binary(msg_env, piece.size(), &bin);
            memcpy(buf, piece.data(), piece.size());
            enif_send(nullptr, &pid, msg_env, bin);

            llama_batch next = llama_batch_get_one(&new_token, 1);
            if (llama_decode(cres->ctx, next) != 0) {
                send_error("failed to decode token");
                break;
            }
        }

        llama_sampler_free(sampler);

        enif_clear_env(msg_env);
        enif_send(nullptr, &pid, msg_env, enif_make_atom(msg_env, "fin"));
        enif_free_env(msg_env);
        enif_release_resource(cres);
    }).detach();

    return make_ok(env, make_atom(env, "streaming"));
}

// ---------------------------------------------------------------------------
// NIF: embeddings(ctx_ref, text) -> {:ok, [[float]]} | {:error, ...}
//   Returns vec-of-vecs for API compatibility with old Rust NIF
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_embeddings(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    CtxRes *cres;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&cres))
        return make_error(env, "bad context reference");

    std::string text;
    if (!get_string(env, argv[1], text))
        return make_error(env, "bad text argument");

    std::lock_guard<std::mutex> lock(cres->mu);

    const llama_vocab *vocab = llama_model_get_vocab(cres->model);
    auto tokens = tokenize(vocab, text, true);

    llama_memory_clear(llama_get_memory(cres->ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    if (llama_decode(cres->ctx, batch) != 0)
        return make_error(env, "decode failed");

    float *embd = llama_get_embeddings(cres->ctx);
    if (!embd)
        return make_error(env, "embeddings not available (context not in embedding mode?)");

    int n_embd = llama_model_n_embd(cres->model);

    // Normalize
    float norm = 0;
    for (int i = 0; i < n_embd; i++) norm += embd[i] * embd[i];
    norm = sqrtf(norm);

    // Build inner list
    ERL_NIF_TERM inner = enif_make_list(env, 0);
    for (int i = n_embd - 1; i >= 0; i--) {
        double val = (norm > 0) ? (double)(embd[i] / norm) : 0.0;
        inner = enif_make_list_cell(env, enif_make_double(env, val), inner);
    }

    // Wrap in outer list for vec-of-vecs compatibility
    ERL_NIF_TERM outer = enif_make_list1(env, inner);
    return make_ok(env, outer);
}

// ---------------------------------------------------------------------------
// NIF: context_deep_copy(ctx_ref) -> {:ok, new_ctx_ref} | {:error, ...}
//   Copies context state including KV cache
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_context_deep_copy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    CtxRes *src;
    if (!enif_get_resource(env, argv[0], CTX_RES_TYPE, (void **)&src))
        return make_error(env, "bad context reference");

    std::lock_guard<std::mutex> lock(src->mu);

    // Create new context from same model with same params
    llama_context_params params = llama_context_default_params();
    params.n_ctx = llama_n_ctx(src->ctx);
    params.n_batch = llama_n_batch(src->ctx);

    llama_context *new_ctx = llama_init_from_model(src->model, params);
    if (!new_ctx)
        return make_error(env, "failed to create context for deep copy");

    // Copy state via serialization
    size_t state_size = llama_state_get_size(src->ctx);
    std::vector<uint8_t> state_buf(state_size);
    size_t written = llama_state_get_data(src->ctx, state_buf.data(), state_buf.size());
    if (written == 0) {
        llama_free(new_ctx);
        return make_error(env, "failed to serialize context state");
    }

    size_t loaded = llama_state_set_data(new_ctx, state_buf.data(), written);
    if (loaded == 0) {
        llama_free(new_ctx);
        return make_error(env, "failed to restore context state");
    }

    auto *cres = static_cast<CtxRes *>(
        enif_alloc_resource(CTX_RES_TYPE, sizeof(CtxRes)));
    new (cres) CtxRes();
    cres->ctx   = new_ctx;
    cres->model = src->model;

    ERL_NIF_TERM term = enif_make_resource(env, cres);
    enif_release_resource(cres);
    return make_ok(env, term);
}

// ---------------------------------------------------------------------------
// NIF: chat_apply_template(model_ref, messages, add_generation_prompt)
// ---------------------------------------------------------------------------
static ERL_NIF_TERM
nif_chat_apply_template(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
    ModelRes *mres;
    if (!enif_get_resource(env, argv[0], MODEL_RES_TYPE, (void **)&mres))
        return make_error(env, "bad model reference");

    unsigned int n_msgs;
    if (!enif_get_list_length(env, argv[1], &n_msgs))
        return make_error(env, "bad messages list");

    bool add_ass = true;
    if (argc > 2) get_bool(env, argv[2], add_ass);

    std::vector<llama_chat_message> msgs(n_msgs);
    std::vector<std::string> roles(n_msgs), contents(n_msgs);

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < n_msgs; i++) {
        enif_get_list_cell(env, tail, &head, &tail);

        ERL_NIF_TERM role_term, content_term;
        if (!enif_get_map_value(env, head, make_atom(env, "role"), &role_term) ||
            !enif_get_map_value(env, head, make_atom(env, "content"), &content_term)) {
            return make_error(env, "messages must have :role and :content keys");
        }

        if (!get_string(env, role_term, roles[i]) ||
            !get_string(env, content_term, contents[i]))
            return make_error(env, "role and content must be strings");

        msgs[i].role = roles[i].c_str();
        msgs[i].content = contents[i].c_str();
    }

    const char *tmpl = llama_model_chat_template(mres->model, nullptr);

    int32_t needed = llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), add_ass, nullptr, 0);
    if (needed < 0)
        return make_error(env, "failed to apply chat template");

    std::vector<char> buf(needed + 1);
    llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass,
                              buf.data(), buf.size());

    std::string result(buf.data(), needed);
    return make_ok(env, make_binary_string(env, result));
}

// ---------------------------------------------------------------------------
// NIF table
// ---------------------------------------------------------------------------
static ErlNifFunc nif_funcs[] = {
    // model
    {"load_model",            1, nif_load_model,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"load_model",            2, nif_load_model,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    // context
    {"create_context",        1, nif_create_context,        0},
    {"create_context",        2, nif_create_context,        0},
    {"context_deep_copy",     1, nif_context_deep_copy,     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    // tokenization
    {"tokenize",              2, nif_tokenize,              0},
    {"tokenize",              3, nif_tokenize,              0},
    {"detokenize",            2, nif_detokenize,            0},
    {"token_to_piece",        2, nif_token_to_piece,        0},
    // vocab info
    {"vocab_bos",             1, nif_vocab_bos,             0},
    {"vocab_eos",             1, nif_vocab_eos,             0},
    {"vocab_eot",             1, nif_vocab_eot,             0},
    {"vocab_nl",              1, nif_vocab_nl,              0},
    {"vocab_fim_pre",         1, nif_vocab_fim_pre,         0},
    {"vocab_fim_mid",         1, nif_vocab_fim_mid,         0},
    {"vocab_fim_suf",         1, nif_vocab_fim_suf,         0},
    // model info
    {"model_info",            1, nif_model_info,            0},
    // inference
    {"completion",            2, nif_completion,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"completion",            3, nif_completion,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"streaming_completion",  3, nif_streaming_completion,  0},
    {"streaming_completion",  4, nif_streaming_completion,  0},
    // embeddings
    {"embeddings",            2, nif_embeddings,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    // chat template
    {"chat_apply_template",   2, nif_chat_apply_template,   0},
    {"chat_apply_template",   3, nif_chat_apply_template,   0},
};

static void quiet_log(enum ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

static int on_load(ErlNifEnv *env, void **, ERL_NIF_TERM) {
    MODEL_RES_TYPE = enif_open_resource_type(
        env, nullptr, "ExLLamaModel", model_res_dtor,
        ERL_NIF_RT_CREATE, nullptr);
    CTX_RES_TYPE = enif_open_resource_type(
        env, nullptr, "ExLLamaCtx", ctx_res_dtor,
        ERL_NIF_RT_CREATE, nullptr);

    llama_log_set(quiet_log, nullptr);

    ggml_backend_load_all();

    return 0;
}

ERL_NIF_INIT(Elixir.ExLLama.Nif, nif_funcs, on_load, NULL, NULL, NULL)
