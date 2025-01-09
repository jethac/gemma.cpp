#define GEMMA_EXPORTS
#include "gemma/c_api.h"
#include "gemma/gemma.h"
#include "gemma/common.h"
#include "util/app.h"
#include "util/threading.h"
#include <memory>

// Internal implementation details
namespace {
std::unique_ptr<gcpp::Gemma> CreateGemmaFromParams(const char* tokenizer_path, const char* model_type, 
                           const char* weights_path, const char* weight_type) {
    // ---- Initialize loader args (model configuration) ----
    gcpp::LoaderArgs loader(tokenizer_path, weights_path, model_type);
    loader.weight_type_str = weight_type;  // Not part of constructor, set separately

    // Validate will parse the model type and weight type strings
    if (const char* error = loader.Validate()) {
        HWY_ABORT("Invalid loader configuration: %s", error);
    }

    // ---- Initialize inference args (generation settings) ----
    gcpp::InferenceArgs inference;
    inference.Init();
    inference.temperature = 0.7f;          // Default temperature
    inference.top_k = 1;                   // Default top_k sampling
    inference.max_generated_tokens = 2048;  // Default max tokens
    inference.deterministic = false;        // Allow sampling
    inference.multiturn = false;           // Single turn by default

    // ---- Initialize app args (runtime settings) ----
    gcpp::AppArgs app;
    app.Init();
    app.max_packages = 1;                  // Limit to single package as in main()
    app.verbosity = 0;                     // Minimal output for API use
    app.spin = gcpp::Tristate::kFalse;     // No spinning for API use

    // ---- Set up thread pools ----
    gcpp::NestedPools pools = gcpp::CreatePools(app);

    // ---- Create and return the model ----
    return gcpp::AllocateGemma(loader, pools);
}

} // anonymous namespace

struct GemmaContext {
    std::unique_ptr<gcpp::Gemma> model;
    std::unique_ptr<gcpp::KVCache> kv_cache;
    std::string prompt_buffer;    // Reusable buffer for prompt text
    std::string result_buffer;    // Reusable buffer for generation results
    std::vector<int> token_buffer;  // Reusable buffer for tokenized input
};

extern "C" {

GEMMA_API GemmaContext* GemmaCreate(
    const char* tokenizer_path,
    const char* model_type,
    const char* weights_path,
    const char* weight_type) {
    // Initialize library global state
    gcpp::InitializeGemmaLibrary();
    
    auto ctx = new GemmaContext();
    try {
        ctx->model = CreateGemmaFromParams(tokenizer_path, model_type, weights_path, weight_type);
        ctx->kv_cache = std::make_unique<gcpp::KVCache>(
            gcpp::KVCache::Create(ctx->model->GetModelConfig(), 2048)); // Default prefill size
        return ctx;
    } catch (...) {
        delete ctx;
        return nullptr;
    }
}

GEMMA_API void GemmaDestroy(GemmaContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output, int max_length) {
    if (!ctx || !ctx->model || !ctx->kv_cache || !prompt || !output || max_length <= 0) return -1;
    
    try {
        std::mt19937 gen;
        gcpp::InferenceArgs inference;
        inference.Init();
        inference.max_generated_tokens = max_length;
        inference.temperature = 0.7f;
        inference.top_k = 1;
        inference.deterministic = false;

        gcpp::AppArgs app;
        app.Init();
        app.max_packages = 1;
        app.spin = gcpp::Tristate::kFalse;
        app.verbosity = 0;

        // Clear and reuse buffers
        ctx->result_buffer.clear();
        ctx->prompt_buffer.assign(prompt);
        ctx->token_buffer.clear();

        auto stream_token = [ctx](int token, float) {
            std::string token_text;
            if (token != gcpp::EOS_ID) {
                if (ctx->model->Tokenizer().Decode(std::vector<int>{token}, &token_text)) {
                    ctx->result_buffer.append(token_text);
                    return true;
                }
            }
            return false;
        };

        gcpp::RuntimeConfig runtime_config = {
            .gen = &gen,
            .verbosity = 0,
            .stream_token = stream_token,
            .use_spinning = app.spin
        };
        inference.CopyTo(runtime_config);

        // Reuse token buffer
        ctx->token_buffer = gcpp::WrapAndTokenize(
            ctx->model->Tokenizer(), ctx->model->Info(), 0, ctx->prompt_buffer);

        gcpp::TimingInfo timing_info = {.verbosity = 0};
        ctx->model->Generate(runtime_config, ctx->token_buffer, 0, 0, *ctx->kv_cache, timing_info);

        if (ctx->result_buffer.length() >= static_cast<size_t>(max_length)) return -1;
        strcpy(output, ctx->result_buffer.c_str());
        return static_cast<int>(ctx->result_buffer.length());
    } catch (...) {
        return -1;
    }
}

}