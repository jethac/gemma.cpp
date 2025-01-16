#include "gemma/context.h"

#include <random>

#include "util/threading.h"

namespace gcpp {

GemmaContext::GemmaContext(const char* tokenizer_path, const char* model_type,
                           const char* weights_path, const char* weight_type,
                           const AppArgs& app_args, int max_length)
    : pools(CreatePools(app_args)) {
  LoaderArgs loader(tokenizer_path, weights_path, model_type);
  loader.weight_type_str = weight_type;

  if (const char* error = loader.Validate()) {
    HWY_ABORT("Invalid loader configuration: %s", error);
  }

  // Initialize cached args
  inference_args.Init();
  inference_args.max_generated_tokens = max_length;
  inference_args.temperature = 0.7f;
  inference_args.top_k = 1;
  inference_args.deterministic = false;

  Allocator::Init(pools.Topology());
  model = AllocateGemma(loader, pools);
  kv_cache =
      std::make_unique<KVCache>(KVCache::Create(model->GetModelConfig(), 2048));
}

int GemmaContext::Generate(const char* prompt, char* output, int max_length) {
  if (!model || !kv_cache || !prompt || !output || max_length <= 0) return -1;

  try {
    // Clear and reuse buffers
    result_buffer.clear();
    prompt_buffer.assign(prompt);
    token_buffer.clear();

    auto stream_token = [this](int token, float) {
      std::string token_text;
      if (token != EOS_ID) {
        if (model->Tokenizer().Decode(std::vector<int>{token}, &token_text)) {
          result_buffer.append(token_text);
          return true;
        }
      }
      return false;
    };

    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = 0,
                                    .stream_token = stream_token,
                                    .use_spinning = Tristate::kFalse};
    inference_args.max_generated_tokens = max_length;
    inference_args.CopyTo(runtime_config);

    token_buffer =
        WrapAndTokenize(model->Tokenizer(), model->Info(), 0, prompt_buffer);

    TimingInfo timing_info = {.verbosity = 0};
    model->Generate(runtime_config, token_buffer, 0, 0, *kv_cache, timing_info);

    if (result_buffer.length() >= static_cast<size_t>(max_length)) return -1;
    strcpy(output, result_buffer.c_str());
    return static_cast<int>(result_buffer.length());
  } catch (...) {
    return -1;
  }
}

}  // namespace gcpp