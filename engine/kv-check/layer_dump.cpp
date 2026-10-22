// LG: engram, live-brain fit. What does his real engine's stream look like going into layer L, word by word?
//   layer-dump MODEL CASES.jsonl OUT LAYER [--lora ADAPTER.gguf] [--cpu]
// CASES, one per line: {"a": the chat up to the moment, "moment": its words} (s0.py's cases.jsonl works). Each case
// is read fresh; the input to layer LAYER (the engine's "l_out-(LAYER-1)") is kept for every word. Writes OUT.f32
// (rows x n_embd, float32, case after case) and OUT.json (per case: rows, and the row where the moment starts), for
// tools/mind/engram/livefit.py to check how well the fragment finder (trained on the raw brain) fits these states.
#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

struct grab {
    std::string name;            // e.g. "l_out-7"
    std::vector<float> rows;     // the last batch's rows, n_embd floats each
    int n_embd = 0;
};

static bool on_tensor(struct ggml_tensor * t, bool ask, void * ud) {
    auto * g = (grab *) ud;
    if (std::strcmp(t->name, g->name.c_str()) != 0) {
        return !ask;             // ask: we don't want it; done: carry on
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || t->ne[0] != g->n_embd) {
        std::fprintf(stderr, "layer-dump: %s is not [n_embd, n_tokens] f32\n", t->name);
        return false;
    }
    const size_t n = (size_t) t->ne[0] * t->ne[1];
    g->rows.resize(n);
    ggml_backend_tensor_get(t, g->rows.data(), 0, n * sizeof(float));
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(std::max(n, 0));
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, false, true);
    return toks;
}

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "see the top of layer_dump.cpp\n"); return 1; }
    const int L = std::atoi(argv[4]);
    const char * lora = nullptr;
    llama_model_params mp = llama_model_default_params();
    for (int i = 5; i < argc; i++) {
        if (!std::strcmp(argv[i], "--lora") && i + 1 < argc) { lora = argv[++i]; }
        if (!std::strcmp(argv[i], "--cpu")) { mp.n_gpu_layers = 0; }   // a big brain that shouldn't crowd the GPU
    }
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    grab g;
    g.name = "l_out-" + std::to_string(L - 1);
    g.n_embd = llama_model_n_embd(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 1024;
    cp.n_ubatch = 1024;          // one pass per case, so one callback holds every word
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &g;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    if (lora) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora);
        float scale = 1.0f;
        if (!ad) { std::fprintf(stderr, "layer-dump: could not load the adapter %s\n", lora); return 1; }
        llama_set_adapters_lora(ctx, &ad, 1, &scale);
    }
    std::ifstream in(argv[2]);
    std::ofstream out(std::string(argv[3]) + ".f32", std::ios::binary);
    json index = json::array();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        const json c = json::parse(line);
        const auto a = tokenize(vocab, c["a"].get<std::string>());
        const auto m = tokenize(vocab, c["moment"].get<std::string>());
        std::vector<llama_token> toks(a);
        toks.insert(toks.end(), m.begin(), m.end());
        if ((int) toks.size() > (int) cp.n_batch) { std::fprintf(stderr, "layer-dump: case too long\n"); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_get_one(toks.data(), (int) toks.size());
        g.rows.clear();
        if (llama_decode(ctx, b) != 0 || g.rows.size() != toks.size() * (size_t) g.n_embd) {
            std::fprintf(stderr, "layer-dump: decode failed or %s not seen (%zu floats)\n", g.name.c_str(),
                         g.rows.size());
            return 1;
        }
        out.write((const char *) g.rows.data(), g.rows.size() * sizeof(float));
        index.push_back({{"rows", toks.size()}, {"moment_from", a.size()}});
    }
    std::ofstream(std::string(argv[3]) + ".json") << json({{"layer", L}, {"n_embd", g.n_embd},
                                                          {"adapter", lora != nullptr}, {"cases", index}}).dump(1);
    std::fprintf(stderr, "layer-dump: %zu cases, %s\n", index.size(), g.name.c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
