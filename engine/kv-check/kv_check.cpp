// LG: engram S0. Can his real engine keep a moment's KV cache and slide it into another chat (short-term memory)?
//   kv-check MODEL CASES.jsonl OUT.jsonl [--no-swa-full] [--offset D] [--lora ADAPTER.gguf]
// CASES, one per line (tools/mind/engram/s0.py writes them): {"a": chat A up to the moment, "moment": its words,
//   "b": chat B up to where the memory goes, "q": the rest of chat B (the question and the start of his turn)}
// 1. shift check (case 0): read A+moment, then move the whole cache D positions later and read q; must give the
//    same next-word scores as reading everything at D from the start. The floor is the gap between reading at 0 and
//    at D (rounding only: his attention only cares about distances).
// 2. restore check (case 0): save the moment's cache, wipe everything, re-read A, put the moment back, read q;
//    must give the same scores as reading A+moment+q straight.
// 3. every case three ways, greedy answer: text (the moment read fresh in chat B), kv (the moment's saved cache
//    from chat A, moved into chat B right before the question), none. One JSON line per case and way to OUT.
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static const int MAX_NEW = 12;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(std::max(n, 0));
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, false, true);
    return toks;
}

// read toks at positions pos0.. into sequence 0; keep next-word scores for every token (all) or only the last
static bool feed(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, bool all) {
    const int nb = (int) llama_n_batch(ctx);
    if (all && (int) toks.size() > nb) { std::fprintf(stderr, "kv-check: scored span longer than a batch\n"); return false; }
    llama_batch b = llama_batch_init(nb, 0, 1);
    bool ok = true;
    for (size_t i = 0; i < toks.size() && ok; i += nb) {
        const int n = (int) std::min<size_t>(nb, toks.size() - i);
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j] = toks[i + j];
            b.pos[j] = pos0 + (int) i + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = all || i + j == toks.size() - 1;
        }
        ok = llama_decode(ctx, b) == 0;
    }
    llama_batch_free(b);
    return ok;
}

static std::vector<float> scores(llama_context * ctx, int n, int n_vocab) {
    std::vector<float> out((size_t) n * n_vocab);
    for (int i = 0; i < n; i++) {
        std::memcpy(out.data() + (size_t) i * n_vocab, llama_get_logits_ith(ctx, i), sizeof(float) * n_vocab);
    }
    return out;
}

// the moment's cells [p0, p1) of sequence 0, saved the way llama.cpp saves a sequence
static std::vector<uint8_t> save_span(llama_context * ctx, int p0, int p1) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, 0, 1, p0, p1);
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 1));
    buf.resize(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 1));
    llama_memory_seq_rm(mem, 1, -1, -1);
    return buf;
}

// put a saved moment back into sequence 0, moved from its old first position to a new one
static bool put_back(llama_context * ctx, const std::vector<uint8_t> & buf, int from, int to) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (llama_state_seq_set_data(ctx, buf.data(), buf.size(), 1) == 0) { return false; }
    if (to != from) { llama_memory_seq_add(mem, 1, -1, -1, to - from); }
    llama_memory_seq_cp(mem, 1, 0, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);
    return true;
}

static std::string answer(llama_context * ctx, const llama_vocab * vocab, int pos, llama_token stop) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::string out;
    for (int k = 0; k < MAX_NEW; k++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        const llama_token t = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        if (t == stop || llama_vocab_is_eog(vocab, t)) { break; }
        char piece[256];
        const int n = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
        out.append(piece, std::max(n, 0));
        if (!feed(ctx, {t}, pos++, false)) { break; }
    }
    return out;
}

struct diff { double max_abs = 0; int same_top = 0, n = 0; };

static diff compare(const std::vector<float> & x, const std::vector<float> & y, int n_vocab) {
    diff d;
    d.n = (int) (x.size() / n_vocab);
    for (int i = 0; i < d.n; i++) {
        const float * a = x.data() + (size_t) i * n_vocab;
        const float * b = y.data() + (size_t) i * n_vocab;
        for (int v = 0; v < n_vocab; v++) { d.max_abs = std::max(d.max_abs, (double) std::fabs(a[v] - b[v])); }
        d.same_top += (std::max_element(a, a + n_vocab) - a) == (std::max_element(b, b + n_vocab) - b);
    }
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "see the top of kv_check.cpp\n"); return 1; }
    bool swa_full = true;
    int D = 100;
    const char * lora = nullptr;
    for (int i = 4; i < argc; i++) {
        if (!std::strcmp(argv[i], "--no-swa-full")) { swa_full = false; }
        if (!std::strcmp(argv[i], "--offset") && i + 1 < argc) { D = std::atoi(argv[++i]); }
        if (!std::strcmp(argv[i], "--lora") && i + 1 < argc) { lora = argv[++i]; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    cp.swa_full = swa_full;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    // his talk adapter (tools/train), worn for everything, as in his thinking context
    if (lora) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora);
        float scale = 1.0f;
        if (!ad) { std::fprintf(stderr, "kv-check: could not load the adapter %s\n", lora); return 1; }
        llama_set_adapters_lora(ctx, &ad, 1, &scale);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    const bool can_shift = llama_memory_can_shift(mem);
    const llama_token stop = tokenize(vocab, "<turn|>").at(0);

    std::vector<json> cases;
    {
        std::ifstream in(argv[2]);
        for (std::string line; std::getline(in, line);) {
            if (!line.empty()) { cases.push_back(json::parse(line)); }
        }
    }
    json head = {{"swa_full", swa_full}, {"can_shift", can_shift}, {"offset", D}, {"adapter", lora != nullptr}};
    std::printf("%s\n", head.dump().c_str());
    if (!can_shift) { std::fprintf(stderr, "kv-check: this cache cannot move words (needs swa_full)\n"); return 2; }

    const auto a = tokenize(vocab, cases[0]["a"].get<std::string>());
    const auto m = tokenize(vocab, cases[0]["moment"].get<std::string>());
    const auto q = tokenize(vocab, cases[0]["q"].get<std::string>());
    std::vector<llama_token> am(a);
    am.insert(am.end(), m.begin(), m.end());
    const int na = (int) a.size(), nam = (int) am.size(), nq = (int) q.size();

    // 1. shift check
    llama_memory_clear(mem, true);
    feed(ctx, am, 0, false);
    feed(ctx, q, nam, true);
    const auto at0 = scores(ctx, nq, n_vocab);
    llama_memory_clear(mem, true);
    feed(ctx, am, D, false);
    feed(ctx, q, nam + D, true);
    const auto atD = scores(ctx, nq, n_vocab);
    llama_memory_clear(mem, true);
    feed(ctx, am, 0, false);
    llama_memory_seq_add(mem, 0, -1, -1, D);
    feed(ctx, q, nam + D, true);
    const auto moved = scores(ctx, nq, n_vocab);
    const diff floor_ = compare(at0, atD, n_vocab), shift = compare(moved, atD, n_vocab);

    // 2. restore check
    llama_memory_clear(mem, true);
    feed(ctx, am, 0, false);
    const auto saved = save_span(ctx, na, nam);
    llama_memory_clear(mem, true);
    feed(ctx, a, 0, false);
    const bool put = put_back(ctx, saved, na, na);
    feed(ctx, q, nam, true);
    const auto restored = scores(ctx, nq, n_vocab);
    const diff restore = compare(restored, at0, n_vocab);
    json checks = {
        {"floor_read_at_0_vs_D", {{"max_abs", floor_.max_abs}, {"same_top", floor_.same_top}, {"n", floor_.n}}},
        {"shift_moved_vs_read_at_D", {{"max_abs", shift.max_abs}, {"same_top", shift.same_top}, {"n", shift.n}}},
        {"restore_vs_straight", {{"max_abs", restore.max_abs}, {"same_top", restore.same_top}, {"n", restore.n},
                                 {"put_back", put}}},
        {"saved_bytes_per_word", (double) saved.size() / m.size()}};
    std::printf("%s\n", checks.dump().c_str());

    // 3. every case, three ways
    std::ofstream out(argv[3]);
    for (size_t i = 0; i < cases.size(); i++) {
        const auto & c = cases[i];
        const auto ca = tokenize(vocab, c["a"].get<std::string>()), cm = tokenize(vocab, c["moment"].get<std::string>());
        const auto cb = tokenize(vocab, c["b"].get<std::string>()), cq = tokenize(vocab, c["q"].get<std::string>());
        for (const std::string way : {"text", "kv", "none"}) {
            llama_memory_clear(mem, true);
            int pos = 0;
            if (way == "kv") {
                std::vector<llama_token> x(ca);
                x.insert(x.end(), cm.begin(), cm.end());
                feed(ctx, x, 0, false);
                const auto buf = save_span(ctx, (int) ca.size(), (int) x.size());
                llama_memory_clear(mem, true);
                feed(ctx, cb, 0, false);
                if (!put_back(ctx, buf, (int) ca.size(), (int) cb.size())) { std::fprintf(stderr, "put back failed\n"); return 3; }
                pos = (int) (cb.size() + cm.size());
                feed(ctx, cq, pos, false);
                pos += (int) cq.size();
            } else {
                std::vector<llama_token> x(cb);
                if (way == "text") { x.insert(x.end(), cm.begin(), cm.end()); }
                x.insert(x.end(), cq.begin(), cq.end());
                feed(ctx, x, 0, false);
                pos = (int) x.size();
            }
            json row = {{"i", i}, {"way", way}, {"said", answer(ctx, vocab, pos, stop)}};
            out << row.dump() << "\n";
        }
        out.flush();
        std::fprintf(stderr, "%zu/%zu cases done\n", i + 1, cases.size());
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
