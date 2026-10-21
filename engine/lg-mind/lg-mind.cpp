// LG: Little Guy's brain for his mind loop: keeps one model loaded and answers one JSON request per stdin line with
// one JSON line on stdout. It does what tools/mind/life_sim.py does with PyTorch hooks, with llama.cpp:
//   {"op":"think", "prompt":<templated text>, "cvec":<path or null>, "max_tokens":N, "temperature":T, "top_p":P,
//    "seed":S}
//       -> {"text": ..., "n_tok": N}. cvec = a raw float32 file of (n_layer-1) x n_embd, row r-1 = the push added to
//       room r's output (llama.cpp's control-vector layout); his feelings as a drip. null = no push.
//       "special": true keeps special tokens in the text (the thinking channel's <|channel>thought ... <channel|>).
//       "think_tokens": K (with special) gives his private thought its own budget: after K words of thinking the
//       thought is closed for him (<channel|>) and max_tokens then counts only his answer, so a long thought can never
//       eat the reply. -> also "n_thought".
//       SIGUSR1 hushes the writing in progress (he heard something): it stops at the next word -> "hushed": true.
//       "choices": ["no<turn|>", ...] instead of writing: the log-probability of each choice as the whole answer
//       (every token of it, teacher-forced, with the same drip) -> {"choice_logprobs": {choice: lp}}; a decision's
//       real odds (the policy his choices are trained on later), not one sampled word.
//       "audio": ["a.wav", ...] (needs --mmproj, his ears): he hears each file where the prompt has the ears' marker
//       (<__media__>), in order; wav, mp3 or flac.
//   {"op":"states", "prompt":<text>, "prefix":<text> (or "skip":K), "rooms":[...], "out":<path>, "mode":"mean"|"all"}
//       -> {"n_tok": T}. Runs the text with no push and writes the outputs of the given rooms for the tokens after the
//       prefix (K = the prefix's length in tokens) to <out>
//       as raw float32: mean = rooms x n_embd, all = rooms x T x n_embd. His mind projects them onto his dials
//       (how a note "reads" inside him, and the natural spread of each dial).
// Thinking and reading use two contexts on the same weights: reading needs an eval callback, which makes the backend
// stop after every captured node, and thinking must not pay for that.
// usage: llama-lg-mind -m <model.gguf> [-c 4096] [-ngl 99] [--lora adapter.gguf] [--mmproj ears.gguf]   (common args)
//        env LG_LISTEN=listener.gguf: his listener (ears round 8), worn only while a sound chunk is read
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// set by SIGUSR1 (the brainstem's /hush, when Immy talks to him mid-daydream); checked between words
static std::atomic<bool> hush{false};
static void on_hush(int) { hush = true; }

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
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) {
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
    // what he has learned (--lora, from tools/train/choices.py) rides on top of his frozen brain, in the thinking
    // context only: how a text reads inside him (the reader) stays the untrained brain his feelings were measured on
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> lora_scales;
    for (const auto & la : params.lora_adapters) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, la.path.c_str());
        if (!ad) {
            std::fprintf(stderr, "lg-mind: could not load the adapter %s\n", la.path.c_str());
            return 1;
        }
        loras.push_back(ad);
        lora_scales.push_back(la.scale);
    }
    if (!loras.empty()) {
        llama_set_adapters_lora(think, loras.data(), loras.size(), lora_scales.data());
    }
    // his listener (ears round 8, env LG_LISTEN, from widen.py listenout): an adapter on his language rooms trained
    // only at the sound's own positions, so it's worn only while he reads a sound chunk and taken off for every word
    // (the words after a sound still read what the listener made of it, through the cache, as in training)
    llama_adapter_lora * listener = nullptr;
    if (const char * lp = std::getenv("LG_LISTEN"); lp && *lp) {
        listener = llama_adapter_lora_init(model, lp);
        if (!listener) {
            std::fprintf(stderr, "lg-mind: could not load the listener %s\n", lp);
            return 1;
        }
    }
    auto listening = [&](bool on) {
        std::vector<llama_adapter_lora *> ads = loras;
        std::vector<float> sc = lora_scales;
        if (on) { ads.push_back(listener); sc.push_back(1.0f); }
        llama_set_adapters_lora(think, ads.data(), ads.size(), sc.data());
    };
    const int n_batch = (int) llama_n_batch(think);
    // his ears (and eyes): the audio tower that came with his brain, in the thinking context only
    mtmd_context * ears = nullptr;
    if (!params.mmproj.path.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = params.mmproj_use_gpu;
        mp.n_threads = params.cpuparams.n_threads;
        mp.print_timings = false;
        ears = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
        if (!ears || !mtmd_support_audio(ears)) {
            std::fprintf(stderr, "lg-mind: could not load the ears %s\n", params.mmproj.path.c_str());
            return 1;
        }
    }
    std::vector<float> cvec;
    std::string cvec_path;

    std::signal(SIGUSR1, on_hush);
    std::cout << json{{"ready", true}, {"n_embd", n_embd}, {"n_layer", n_layer}, {"ears", ears != nullptr}}.dump() << std::endl;
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
                llama_pos n_prompt = 0;
                if (req.contains("audio") && !req["audio"].empty()) {
                    if (!ears) { throw std::runtime_error("audio given but no ears loaded (--mmproj)"); }
                    std::vector<mtmd_helper_bitmap_wrapper> sounds;
                    std::vector<const mtmd_bitmap *> ptrs;
                    auto free_sounds = [&] { for (auto & w : sounds) { mtmd_bitmap_free(w.bitmap); } };
                    for (const auto & a : req["audio"]) {
                        auto w = mtmd_helper_bitmap_init_from_file(ears, a.get<std::string>().c_str(), false);
                        if (!w.bitmap) { free_sounds(); throw std::runtime_error("cannot read audio " + a.get<std::string>()); }
                        sounds.push_back(w);
                        ptrs.push_back(w.bitmap);
                    }
                    const std::string prompt = req.at("prompt").get<std::string>();
                    mtmd_input_text txt{prompt.c_str(), /*add_special=*/false, /*parse_special=*/true};
                    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
                    const int32_t tr = mtmd_tokenize(ears, chunks, &txt, ptrs.data(), ptrs.size());
                    free_sounds();
                    if (tr != 0) {
                        mtmd_input_chunks_free(chunks);
                        throw std::runtime_error("the prompt's markers don't match the sounds (" + std::to_string(tr) + ")");
                    }
                    // chunk by chunk (what mtmd_helper_eval_chunks does), so the listener is on for the sound only
                    int32_t er = 0;
                    const size_t n_chunks = mtmd_input_chunks_size(chunks);
                    for (size_t i = 0; i < n_chunks && er == 0; i++) {
                        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
                        const bool sound = listener && mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO;
                        if (sound) { listening(true); }
                        er = mtmd_helper_eval_chunk_single(ears, think, ch, n_prompt, 0, n_batch, i == n_chunks - 1, &n_prompt);
                        if (sound) { listening(false); }
                    }
                    mtmd_input_chunks_free(chunks);
                    if (er != 0) { throw std::runtime_error("decode failed (prompt with sound)"); }
                } else {
                    const auto toks = tokenize(vocab, req.at("prompt").get<std::string>());
                    if (!eval(think, toks, n_batch)) { throw std::runtime_error("decode failed (prompt)"); }
                    n_prompt = (llama_pos) toks.size();
                }
                if (req.contains("choices")) {
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    auto log_softmax_at = [&](const float * lg, llama_token t) {
                        float mx = lg[0];
                        for (int k = 1; k < n_vocab; k++) { mx = std::max(mx, lg[k]); }
                        double z = 0.0;
                        for (int k = 0; k < n_vocab; k++) { z += std::exp((double) lg[k] - mx); }
                        return (double) lg[t] - mx - std::log(z);
                    };
                    const std::vector<float> first(llama_get_logits_ith(think, -1), llama_get_logits_ith(think, -1) + n_vocab);
                    json lp = json::object();
                    for (const auto & c : req["choices"]) {
                        const auto ct = tokenize(vocab, c.get<std::string>());
                        double sum = 0.0;
                        for (size_t j = 0; j < ct.size(); j++) {
                            sum += log_softmax_at(j == 0 ? first.data() : llama_get_logits_ith(think, -1), ct[j]);
                            if (j + 1 < ct.size()) {
                                llama_token t = ct[j];
                                if (llama_decode(think, llama_batch_get_one(&t, 1)) != 0) {
                                    throw std::runtime_error("decode failed (choice)");
                                }
                            }
                        }
                        llama_memory_seq_rm(llama_get_memory(think), 0, n_prompt, -1);
                        lp[c.get<std::string>()] = sum;
                    }
                    out = {{"choice_logprobs", lp}};
                    std::cout << out.dump() << std::endl;
                    continue;
                }
                const bool special = req.value("special", false);
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
                const int think_tokens = special ? req.value("think_tokens", 0) : 0;
                static const std::string OPEN = "<|channel>thought", CLOSE = "<channel|>";
                // still inside his private thought: it opened and hasn't closed yet
                auto in_thought = [&] { return think_tokens > 0 && text.rfind(OPEN, 0) == 0 && text.find(CLOSE) == std::string::npos; };
                int n_tok = 0, n_thought = 0, n_reply = 0;
                bool hushed = false;
                // a hush is only for the writing it interrupts: one left over from before must not cut the next
                // (the brainstem only sends one while a daydream is being written)
                hush = false;
                while ((think_tokens > 0 ? n_reply : n_tok) < max_tokens) {
                    if (hush) { hushed = true; break; }
                    llama_token t = llama_sampler_sample(smpl, think, -1);
                    if (llama_vocab_is_eog(vocab, t)) { break; }
                    char buf[256];
                    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, special);
                    const bool was_thought = in_thought() || (think_tokens > 0 && text.empty());
                    if (n > 0) { text.append(buf, n); }
                    if (llama_decode(think, llama_batch_get_one(&t, 1)) != 0) {
                        llama_sampler_free(smpl);
                        throw std::runtime_error("decode failed (reply)");
                    }
                    n_tok++;
                    (was_thought ? n_thought : n_reply)++;
                    if (in_thought() && n_thought >= think_tokens) {        // enough thinking: he closes the thought
                        auto ct = tokenize(vocab, CLOSE);
                        if (!eval(think, ct, n_batch)) {
                            llama_sampler_free(smpl);
                            throw std::runtime_error("decode failed (closing the thought)");
                        }
                        text += CLOSE;
                    }
                    if (n_tok >= max_tokens + think_tokens + 64) { break; }   // a hard ceiling, whatever happens
                }
                llama_sampler_free(smpl);
                out = {{"text", text}, {"n_tok", n_tok}};
                if (think_tokens > 0) { out["n_thought"] = n_thought; }
                if (hushed) { out["hushed"] = true; }
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
    if (ears) { mtmd_free(ears); }
    llama_free(reader);
    llama_free(think);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
