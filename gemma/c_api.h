#ifndef GEMMA_C_API_H_
#define GEMMA_C_API_H_

#ifdef _WIN32
    #ifdef GEMMA_EXPORTS
        #define GEMMA_API __declspec(dllexport)
    #else
        #define GEMMA_API __declspec(dllimport)
    #endif
#else
    #define GEMMA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GemmaContext GemmaContext;

GEMMA_API GemmaContext* GemmaCreate(
    const char* tokenizer_path,
    const char* model_type,
    const char* weights_path,
    const char* weight_type);
GEMMA_API void GemmaDestroy(GemmaContext* ctx);
GEMMA_API int GemmaGenerate(GemmaContext* ctx, const char* prompt, char* output, int max_length);

#ifdef __cplusplus
}
#endif

#endif  // GEMMA_C_API_H_