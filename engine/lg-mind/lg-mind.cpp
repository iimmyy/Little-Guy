// LG: Little Guy's brain for his mind loop: keeps one model loaded and answers one JSON request per stdin line with
// one JSON line on stdout. It does what tools/mind/life_sim.py does with PyTorch hooks, with llama.cpp:
//   {"op":"think", "prompt":<templated text>, "cvec":<path or null>, "max_tokens":N, "temperature":T, "top_p":P,
//    "seed":S}
//       -> {"text": ...}. cvec = a raw float32 file of (n_layer-1) x n_embd, row r-1 = the push added to room r's
//       output (llama.cpp's control-vector layout); his feelings as a drip. null = no push.
//   {"op":"states", "prompt":<text>, "prefix":<text> (or "skip":K), "rooms":[...], "out":<path>, "mode":"mean"|"all"}
//       -> {"n_tok": T}. Runs the text with no push and writes the outputs of the given rooms for the tokens after the
//       prefix (K = the prefix's length in tokens) to <out>
//       as raw float32: mean = rooms x n_embd, all = rooms x T x n_embd. His mind projects them onto his dials
//       (how a note "reads" inside him, and the natural spread of each dial).
// Thinking and reading use two contexts on the same weights: reading needs an eval callback, which makes the backend
// stop after every captured node, and thinking must not pay for that.
// usage: llama-lg-mind -m <model.gguf> [-c 4096] [-ngl 99]   (common args)
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct capture {
    std::map<int, std::vector<float>> rows;   // room -> outputs of every token so far (n_tok x n_embd)
    std::map<int, int> seen;                  // room -> tokens received so far
    int n_embd = 0;
};

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (capture *) user_data;
    int room = -1;
    if (std::sscanf(t->name, "l_out-%d", &room) != 1 || !c->rows.count(room)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != c->n_embd) {
        std::fprintf(stderr, "lg-mind: %s is not a contiguous f32 [n_embd, tokens] tensor\n", t->name);
        return true;
    }
    auto & r = c->rows[room];
    const size_t n = ggml_nelements(t);
    const size_t at = r.size();
    r.resize(at + n);
    ggml_backend_tensor_get(t, r.data() + at, 0, n * sizeof(float));
    c->seen[room] += (int) t->ne[1];
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, false, true);
    return toks;
}

static bool eval(llama_context * ctx, const std::vector<llama_token> & toks, int n_batch) {
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        const int n = (int) std::min<size_t>(n_batch, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks.data()) + i, n)) != 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 4096;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (!model) {
        std::fprintf(stderr, "lg-mind: could not load %s\n", params.model.path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);

    llama_context_params cp = common_context_params_to_llama(params);
    llama_context * think = llama_init_from_model(model, cp);
    capture cap;
    cap.n_embd = n_embd;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &cap;
    llama_context * reader = llama_init_from_model(model, cp);
    if (!think || !reader) {
        std::fprintf(stderr, "lg-mind: could not make the contexts\n");
        return 1;
    }
    const int n_batch = (int) llama_n_batch(think);
    std::vector<float> cvec;
    std::string cvec_path;

    std::cout << json{{"ready", true}, {"n_embd", n_embd}, {"n_layer", n_layer}}.dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        json out;
        try {
            const json req = json::parse(line);
            const std::string op = req.value("op", "");
            if (op == "think") {
                const std::string path = req.contains("cvec") && !req["cvec"].is_null() ? req["cvec"].get<std::string>() : "";
                if (path.empty()) {
                    llama_set_adapter_cvec(think, nullptr, 0, n_embd, 0, 0);
                    cvec_path.clear();
                } else if (path != cvec_path) {
                    FILE * f = std::fopen(path.c_str(), "rb");
                    if (!f) { throw std::runtime_error("cannot open cvec " + path); }
                    cvec.assign((size_t) (n_layer - 1) * n_embd, 0.0f);
                    const size_t got = std::fread(cvec.data(), sizeof(float), cvec.size(), f);
                    std::fclose(f);
                    if (got != cvec.size()) { throw std::runtime_error("cvec has the wrong size"); }
                    if (llama_set_adapter_cvec(think, cvec.data(), cvec.size(), n_embd, 1, n_layer - 1) != 0) {
                        throw std::runtime_error("llama_set_adapter_cvec failed");
                    }
                    cvec_path = path;
                }
                llama_memory_clear(llama_get_memory(think), true);
                const auto toks = tokenize(vocab, req.at("prompt").get<std::string>());
                if (!eval(think, toks, n_batch)) { throw std::runtime_error("decode failed (prompt)"); }
                const float temp = req.value("temperature", 0.0f);
                llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
                if (temp <= 0.0f) {
                    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
                } else {
                    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(req.value("top_p", 1.0f), 1));
                    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
                    llama_sampler_chain_add(smpl, llama_sampler_init_dist(req.value("seed", 0u)));
                }
                std::string text;
                const int max_tokens = req.value("max_tokens", 200);
                for (int i = 0; i < max_tokens; i++) {
                    llama_token t = llama_sampler_sample(smpl, think, -1);
                    if (llama_vocab_is_eog(vocab, t)) { break; }
                    char buf[256];
                    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
                    if (n > 0) { text.append(buf, n); }
                    if (llama_decode(think, llama_batch_get_one(&t, 1)) != 0) {
                        llama_sampler_free(smpl);
                        throw std::runtime_error("decode failed (reply)");
                    }
                }
                llama_sampler_free(smpl);
                out = {{"text", text}};
            } else if (op == "states") {
                cap.rows.clear();
                cap.seen.clear();
                for (int r : req.at("rooms")) {
                    if (r < 0 || r >= n_layer) { throw std::runtime_error("room out of range"); }
                    cap.rows[r] = {};
                    cap.seen[r] = 0;
                }
                llama_memory_clear(llama_get_memory(reader), true);
                const auto toks = tokenize(vocab, req.at("prompt").get<std::string>());
                if (!eval(reader, toks, n_batch)) { throw std::runtime_error("decode failed (read)"); }
                const int skip = req.contains("prefix") ? (int) tokenize(vocab, req["prefix"].get<std::string>()).size()
                                                        : req.value("skip", 0);
                const int T = (int) toks.size() - skip;
                if (T <= 0) { throw std::runtime_error("nothing left after skip"); }
                const bool mean = req.value("mode", "mean") == "mean";
                FILE * f = std::fopen(req.at("out").get<std::string>().c_str(), "wb");
                if (!f) { throw std::runtime_error("cannot write out"); }
                for (auto & [room, r] : cap.rows) {
                    if (cap.seen[room] != (int) toks.size()) {
                        std::fclose(f);
                        throw std::runtime_error("room " + std::to_string(room) + " saw " + std::to_string(cap.seen[room]) +
                                                 " of " + std::to_string(toks.size()) + " tokens");
                    }
                    const float * from = r.data() + (size_t) skip * n_embd;
                    if (mean) {
                        std::vector<double> acc(n_embd, 0.0);
                        for (int t = 0; t < T; t++) { for (int k = 0; k < n_embd; k++) { acc[k] += from[(size_t) t * n_embd + k]; } }
                        std::vector<float> m(n_embd);
                        for (int k = 0; k < n_embd; k++) { m[k] = (float) (acc[k] / T); }
                        std::fwrite(m.data(), sizeof(float), n_embd, f);
                    } else {
                        std::fwrite(from, sizeof(float), (size_t) T * n_embd, f);
                    }
                }
                std::fclose(f);
                out = {{"n_tok", T}};
            } else {
                throw std::runtime_error("unknown op '" + op + "'");
            }
        } catch (const std::exception & e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
    llama_free(reader);
    llama_free(think);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
