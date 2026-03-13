//
// Standalone C++ test for llama.cpp NIF core logic.
// Tests the llama.cpp API directly (no Erlang VM needed).
//
// Build:  make test
// Run:    ./priv/test_llama_nif priv/models/local_llama/tiny_llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
//

#include <llama.h>
#include <ggml.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <regex>

// ---------------------------------------------------------------------------
// Helpers (mirror the NIF helpers)
// ---------------------------------------------------------------------------
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

static std::string detokenize(const llama_vocab *vocab,
                               const std::vector<llama_token> &tokens) {
    std::string out;
    for (auto t : tokens) out += token_to_piece(vocab, t);
    return out;
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(llama_model *model, const llama_vocab *vocab); \
    static struct Register_##name { \
        Register_##name() { } \
    } reg_##name; \
    static void run_##name(llama_model *model, const llama_vocab *vocab) { \
        tests_run++; \
        printf("  test %-40s ", #name); \
        fflush(stdout); \
        try { \
            test_##name(model, vocab); \
            tests_passed++; \
            printf("PASS\n"); \
        } catch (const std::exception &e) { \
            printf("FAIL: %s\n", e.what()); \
        } catch (...) { \
            printf("FAIL: unknown exception\n"); \
        } \
    } \
    static void test_##name(llama_model *model, const llama_vocab *vocab)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        throw std::runtime_error(std::string("assertion failed: ") + #cond \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        throw std::runtime_error(std::string("assertion failed: ") + #a + " == " + #b \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(special_tokens) {
    llama_token bos = llama_vocab_bos(vocab);
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);
    llama_token nl  = llama_vocab_nl(vocab);

    ASSERT(bos >= 0);
    ASSERT(eos >= 0);
    // eot may be -1 for models without it
    (void)eot;
    ASSERT(nl >= 0);

    // BOS and EOS should be different
    ASSERT(bos != eos);
}

TEST(tokenize_basic) {
    auto tokens = tokenize(vocab, "Hello world", true);
    ASSERT(tokens.size() > 0);
    ASSERT(tokens.size() < 20); // sanity

    // First token should be BOS when add_special=true
    ASSERT_EQ(tokens[0], llama_vocab_bos(vocab));
}

TEST(tokenize_no_special) {
    auto with_special = tokenize(vocab, "Hello", true);
    auto without_special = tokenize(vocab, "Hello", false);

    // With special should have BOS prepended
    ASSERT(with_special.size() == without_special.size() + 1);
    ASSERT_EQ(with_special[0], llama_vocab_bos(vocab));
}

TEST(detokenize_roundtrip) {
    std::string input = "The quick brown fox";
    auto tokens = tokenize(vocab, input, false);
    std::string output = detokenize(vocab, tokens);

    // Should roundtrip (may have leading space due to SPM tokenizer)
    ASSERT(output.find("quick brown fox") != std::string::npos);
}

TEST(token_to_piece_bos) {
    llama_token bos = llama_vocab_bos(vocab);
    std::string piece = token_to_piece(vocab, bos);
    // BOS token should produce something (model-dependent)
    ASSERT(piece.size() > 0);
}

TEST(model_info) {
    ASSERT(llama_model_n_embd(model) > 0);
    ASSERT(llama_model_n_ctx_train(model) > 0);
    ASSERT(llama_model_n_layer(model) > 0);
    ASSERT(llama_vocab_n_tokens(vocab) > 0);
    ASSERT(llama_model_size(model) > 0);
    ASSERT(llama_model_n_params(model) > 0);
}

TEST(context_create_destroy) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 64;

    llama_context *ctx = llama_init_from_model(model, params);
    ASSERT(ctx != nullptr);
    ASSERT_EQ(llama_n_ctx(ctx), 512u);
    llama_free(ctx);
}

TEST(completion_basic) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    llama_context *ctx = llama_init_from_model(model, params);
    ASSERT(ctx != nullptr);

    auto tokens = tokenize(vocab, "Hello, my name is", true);
    ASSERT(tokens.size() > 0);

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    int rc = llama_decode(ctx, batch);
    ASSERT_EQ(rc, 0);

    // Set up sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    // Generate a few tokens
    std::string result;
    for (int i = 0; i < 10; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;
        result += token_to_piece(vocab, new_token);

        llama_batch next = llama_batch_get_one(&new_token, 1);
        rc = llama_decode(ctx, next);
        ASSERT_EQ(rc, 0);
    }

    ASSERT(result.size() > 0);

    llama_sampler_free(sampler);
    llama_free(ctx);
}

TEST(completion_deterministic) {
    // Same seed should produce same output
    auto run = [&](unsigned int seed) -> std::string {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 512;
        llama_context *ctx = llama_init_from_model(model, params);

        auto tokens = tokenize(vocab, "Once upon a time", true);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        llama_decode(ctx, batch);

        auto sparams = llama_sampler_chain_default_params();
        llama_sampler *sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

        std::string result;
        for (int i = 0; i < 20; i++) {
            llama_token t = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, t)) break;
            result += token_to_piece(vocab, t);
            llama_batch next = llama_batch_get_one(&t, 1);
            llama_decode(ctx, next);
        }

        llama_sampler_free(sampler);
        llama_free(ctx);
        return result;
    };

    std::string r1 = run(42);
    std::string r2 = run(42);
    std::string r3 = run(99);

    ASSERT_EQ(r1, r2);       // same seed = same output
    ASSERT(r1 != r3);        // different seed = different output (probabilistically)
}

TEST(stop_regex) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    llama_context *ctx = llama_init_from_model(model, params);

    auto tokens = tokenize(vocab, "Hello, my name is", true);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx, batch);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    std::regex stop_regex("\\.");
    std::string result;
    bool stopped = false;

    for (int i = 0; i < 100; i++) {
        llama_token t = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        result += token_to_piece(vocab, t);

        std::smatch match;
        if (std::regex_search(result, match, stop_regex)) {
            result = result.substr(0, match.position() + match.length());
            stopped = true;
            break;
        }

        llama_batch next = llama_batch_get_one(&t, 1);
        llama_decode(ctx, next);
    }

    // Should have stopped at first period
    ASSERT(stopped);
    ASSERT(result.back() == '.');

    llama_sampler_free(sampler);
    llama_free(ctx);
}

TEST(state_deep_copy) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    llama_context *ctx = llama_init_from_model(model, params);

    // Feed some tokens to build KV cache state
    auto tokens = tokenize(vocab, "The capital of France is", true);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx, batch);

    // Save state
    size_t state_size = llama_state_get_size(ctx);
    ASSERT(state_size > 0);

    std::vector<uint8_t> state_buf(state_size);
    size_t written = llama_state_get_data(ctx, state_buf.data(), state_buf.size());
    ASSERT(written > 0);

    // Create new context and restore
    llama_context *ctx2 = llama_init_from_model(model, params);
    ASSERT(ctx2 != nullptr);

    size_t loaded = llama_state_set_data(ctx2, state_buf.data(), written);
    ASSERT(loaded > 0);

    // After restoring state, decode one more token on ctx2
    // to regenerate logits (state save doesn't preserve logits)
    llama_token last_tok = tokens.back();
    llama_batch one_tok = llama_batch_get_one(&last_tok, 1);
    int rc2 = llama_decode(ctx2, one_tok);
    ASSERT_EQ(rc2, 0);

    // Sample from restored context should work
    auto sp = llama_sampler_chain_default_params();
    llama_sampler *s = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    llama_token t = llama_sampler_sample(s, ctx2, -1);
    ASSERT(!llama_vocab_is_eog(vocab, t)); // should get a real token
    llama_sampler_free(s);

    llama_free(ctx);
    llama_free(ctx2);
}

TEST(chat_template) {
    const char *tmpl = llama_model_chat_template(model, nullptr);
    // TinyLlama has a chat template
    ASSERT(tmpl != nullptr);

    llama_chat_message msgs[] = {
        {"user", "Hello!"},
    };

    int32_t needed = llama_chat_apply_template(tmpl, msgs, 1, true, nullptr, 0);
    ASSERT(needed > 0);

    std::vector<char> buf(needed + 1);
    llama_chat_apply_template(tmpl, msgs, 1, true, buf.data(), buf.size());
    std::string result(buf.data(), needed);

    ASSERT(result.find("Hello!") != std::string::npos);
    ASSERT(result.find("user") != std::string::npos);
}

TEST(memory_clear) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    llama_context *ctx = llama_init_from_model(model, params);

    auto tokens = tokenize(vocab, "test", true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx, batch);

    // Clear should not crash
    llama_memory_clear(llama_get_memory(ctx), true);

    // Should be able to decode again after clear
    int rc = llama_decode(ctx, batch);
    ASSERT_EQ(rc, 0);

    llama_free(ctx);
}

TEST(eog_detection) {
    llama_token eos = llama_vocab_eos(vocab);
    ASSERT(llama_vocab_is_eog(vocab, eos));

    llama_token bos = llama_vocab_bos(vocab);
    ASSERT(!llama_vocab_is_eog(vocab, bos));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // Suppress non-error logs
    llama_log_set([](enum ggml_log_level level, const char *text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);

    ggml_backend_load_all();

    printf("Loading model: %s\n", argv[1]);
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    printf("Running C tests...\n\n");

    run_special_tokens(model, vocab);
    run_tokenize_basic(model, vocab);
    run_tokenize_no_special(model, vocab);
    run_detokenize_roundtrip(model, vocab);
    run_token_to_piece_bos(model, vocab);
    run_model_info(model, vocab);
    run_context_create_destroy(model, vocab);
    run_completion_basic(model, vocab);
    run_completion_deterministic(model, vocab);
    run_stop_regex(model, vocab);
    run_state_deep_copy(model, vocab);
    run_chat_template(model, vocab);
    run_memory_clear(model, vocab);
    run_eog_detection(model, vocab);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);

    llama_model_free(model);
    return (tests_passed == tests_run) ? 0 : 1;
}
