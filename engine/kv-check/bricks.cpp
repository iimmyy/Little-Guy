// LG: engram bricks in his real engine. A memory kept as layer-L pieces + name tags is rebuilt inside a new chat
// (rung0.py does it in HF; this is the same splice in llama.cpp, from outside the library: the graph callback).
//   bricks MODEL CASES.jsonl BRICKS.f32 OUT.jsonl [--layer L] [--lora ADAPTER.gguf] [--max-new N]
// CASES (tools/mind/engram/livebricks.py writes them), one per line:
//   {"parts": [{"text": ...} | {"slot": the memory's words or null, "n": words, "brick": its first row in BRICKS}],
//    "cands": [...], "meta": {...}}                       a case: greedy answer + every candidate's score
//   {"check": true, "parts": [{"text": ...}, {"slot": ..., "n": ...}, {"text": ...}]}     the splice check
// Every part is read as its own tokens. A slot holds the memory's words (they bring his name tags, which come from
// the word alone) or n pad tokens (no tags). Its stream going into layer L becomes its BRICKS rows (float32, n_embd
// each), and no layer below L can see it: its cells are hidden in the attention masks of layers 0..L-1, every
// decode, as rung0.py's block. One JSON line per case to OUT, with "meta" copied over.
// The check reads a memory in its own chat three ways: plainly; with the slot hidden below L and the plain reading's
// own layer-L stream put back from the slot on (must give the same next-word scores: the splice lands, the hiding
// stops at L); hidden with nothing put back (must not: the hiding works).
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

static int MAX_NEW = 12;

struct hook {
    std::string name;                                   // "l_out-(L-1)": the stream going into layer L
    int L = 8, n_embd = 0;
    std::vector<std::pair<int, int>> hide;              // cells [from, to) layers below L must not see
    std::map<int, const float *> put;                   // position -> stream row to put in (this decode)
    int pos0 = 0;                                       // position of this decode's first token
    std::vector<float> * grab = nullptr;                // keep this decode's rows going into layer L
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> hidden;   // masks hidden so far, original bytes
};

static int layer_of(const char * name) {
    const char * d = std::strrchr(name, '-');
    return d ? std::atoi(d + 1) : -1;
}

static void hide_cells(hook * h, ggml_tensor * m) {
    std::vector<uint8_t> was(ggml_nbytes(m));
    ggml_backend_tensor_get(m, was.data(), 0, was.size());
    std::vector<uint8_t> now(was);
    const ggml_fp16_t inf16 = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t s = 0; s < m->ne[3]; s++) {
        for (int64_t r = 0; r < m->ne[1]; r++) {
            uint8_t * row = now.data() + s * m->nb[3] + r * m->nb[1];
            for (const auto & [a, b] : h->hide) {
                for (int c = a; c < b && c < m->ne[0]; c++) {
                    if (m->type == GGML_TYPE_F16) { ((ggml_fp16_t *) row)[c] = inf16; }
                    else { ((float *) row)[c] = -INFINITY; }
                }
            }
        }
    }
    ggml_backend_tensor_set(m, now.data(), 0, now.size());
    h->hidden.emplace_back(m, std::move(was));
}

static bool on_tensor(ggml_tensor * t, bool ask, void * ud) {
    auto * h = (hook *) ud;
    const bool mine = h->name == t->name;
    if (ask) {
        // an attention below L is about to run: hide the cells in the mask it reads (once per mask per decode)
        const bool attn = t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SOFT_MAX;
        if (attn && !h->hide.empty() && layer_of(t->name) >= 0 && layer_of(t->name) < h->L) {
            ggml_tensor * m = t->src[t->op == GGML_OP_FLASH_ATTN_EXT ? 3 : 1];
            const bool seen = std::any_of(h->hidden.begin(), h->hidden.end(), [&](auto & x) { return x.first == m; });
            if (m && !seen) {
                if (m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_F32) {
                    std::fprintf(stderr, "bricks: mask of %s is neither f16 nor f32\n", t->name);
                    return true;
                }
                hide_cells(h, m);
            }
        }
        return mine;
    }
    // layers 0..L-1 are done: layers L.. see everything again, and the slot's stream becomes its bricks
    for (auto & [m, was] : h->hidden) { ggml_backend_tensor_set(m, was.data(), 0, was.size()); }
    h->hidden.clear();
    if (t->type != GGML_TYPE_F32 || t->ne[0] != h->n_embd || !ggml_is_contiguous(t)) {
        std::fprintf(stderr, "bricks: %s is not a plain [n_embd, n_tokens] f32\n", t->name);
        return false;
    }
    const size_t row = (size_t) h->n_embd * sizeof(float);
    if (h->grab) {
        h->grab->resize((size_t) t->ne[0] * t->ne[1]);
        ggml_backend_tensor_get(t, h->grab->data(), 0, h->grab->size() * sizeof(float));
    }
    for (int r = 0; r < t->ne[1]; r++) {
        const auto it = h->put.find(h->pos0 + r);
        if (it != h->put.end()) { ggml_backend_tensor_set(t, it->second, r * t->nb[1], row); }
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(std::max(n, 0));
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, false, true);
    return toks;
}

// read toks at positions pos0.. in one decode; next-word scores for every token (all) or the last
static bool feed(llama_context * ctx, hook & h, const std::vector<llama_token> & toks, int pos0, bool all) {
    if ((int) toks.size() > (int) llama_n_ubatch(ctx)) { std::fprintf(stderr, "bricks: too long for one pass\n"); return false; }
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    b.n_tokens = (int) toks.size();
    for (int j = 0; j < b.n_tokens; j++) {
        b.token[j] = toks[j];
        b.pos[j] = pos0 + j;
        b.n_seq_id[j] = 1;
        b.seq_id[j][0] = 0;
        b.logits[j] = all || j == b.n_tokens - 1;
    }
    h.pos0 = pos0;
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

static std::vector<float> last_scores(llama_context * ctx, int n_vocab) {
    const float * lg = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(lg, lg + n_vocab);
}

static double log_softmax_at(const float * lg, int n_vocab, llama_token t) {
    const float mx = *std::max_element(lg, lg + n_vocab);
    double z = 0;
    for (int v = 0; v < n_vocab; v++) { z += std::exp((double) lg[v] - mx); }
    return (double) lg[t] - mx - std::log(z);
}

struct prompt {
    std::vector<llama_token> toks;
    std::vector<std::pair<int, int>> slots;             // positions [from, to) of each slot
    std::map<int, const float *> put;
};

// a case's parts as tokens; slots filled with their bricks (or, for the check, not)
static bool build(const json & c, const llama_vocab * vocab, const std::vector<float> & bricks, int n_embd,
                  prompt & p) {
    for (const auto & part : c["parts"]) {
        if (part.contains("text")) {
            const auto t = tokenize(vocab, part["text"].get<std::string>());
            p.toks.insert(p.toks.end(), t.begin(), t.end());
            continue;
        }
        const int n = part["n"].get<int>();
        std::vector<llama_token> t = part["slot"].is_null() ? std::vector<llama_token>(n, llama_vocab_pad(vocab))
                                                            : tokenize(vocab, part["slot"].get<std::string>());
        if ((int) t.size() != n) {
            std::fprintf(stderr, "bricks: slot reads as %zu words, its bricks have %d\n", t.size(), n);
            return false;
        }
        const int s = (int) p.toks.size();
        p.slots.emplace_back(s, s + n);
        if (part.contains("brick")) {
            const size_t r0 = part["brick"].get<size_t>();
            if ((r0 + n) * n_embd > bricks.size()) { std::fprintf(stderr, "bricks: brick rows out of range\n"); return false; }
            for (int j = 0; j < n; j++) { p.put[s + j] = bricks.data() + (r0 + j) * n_embd; }
        }
        p.toks.insert(p.toks.end(), t.begin(), t.end());
    }
    return true;
}

static void fresh(llama_context * ctx, hook & h) {
    llama_memory_clear(llama_get_memory(ctx), true);
    h.hide.clear();
    h.put.clear();
    h.grab = nullptr;
}

static json check(llama_context * ctx, hook & h, const prompt & p, int n_vocab) {
    std::vector<float> rows;
    fresh(ctx, h);
    h.grab = &rows;
    feed(ctx, h, p.toks, 0, false);
    const auto ref = last_scores(ctx, n_vocab);
    const int s = p.slots.at(0).first;
    auto run = [&](bool put_back) {
        fresh(ctx, h);
        h.hide = p.slots;
        if (put_back) {
            for (int q = s; q < (int) p.toks.size(); q++) { h.put[q] = rows.data() + (size_t) q * h.n_embd; }
        }
        feed(ctx, h, p.toks, 0, false);
        const auto got = last_scores(ctx, n_vocab);
        double d = 0;
        for (int v = 0; v < n_vocab; v++) { d = std::max(d, (double) std::fabs(got[v] - ref[v])); }
        const bool same_top = std::max_element(got.begin(), got.end()) - got.begin() ==
                              std::max_element(ref.begin(), ref.end()) - ref.begin();
        return json({{"max_diff", d}, {"same_top", same_top}});
    };
    json out = {{"hidden_stream_put_back", run(true)}, {"hidden_nothing_put_back", run(false)}};
    fresh(ctx, h);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "see the top of bricks.cpp\n"); return 1; }
    hook h;
    const char * lora = nullptr;
    for (int i = 5; i < argc; i++) {
        if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) { h.L = std::atoi(argv[++i]); }
        if (!std::strcmp(argv[i], "--lora") && i + 1 < argc) { lora = argv[++i]; }
        if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) { MAX_NEW = std::atoi(argv[++i]); }
    }
    h.name = "l_out-" + std::to_string(h.L - 1);
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    h.n_embd = llama_model_n_embd(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 1024;
    cp.n_ubatch = 1024;          // a whole prompt in one pass, so positions map straight to rows
    cp.n_seq_max = 1;
    cp.swa_full = true;          // sliding-window cells = positions, like the full ones
    cp.kv_unified = true;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &h;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    if (lora) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora);
        float scale = 1.0f;
        if (!ad) { std::fprintf(stderr, "bricks: could not load the adapter %s\n", lora); return 1; }
        llama_set_adapters_lora(ctx, &ad, 1, &scale);
    }
    std::vector<float> bricks;
    {
        std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
        bricks.resize((size_t) f.tellg() / sizeof(float));
        f.seekg(0);
        f.read((char *) bricks.data(), bricks.size() * sizeof(float));
    }
    const auto turn_end = tokenize(vocab, "<turn|>");
    const llama_token stop = turn_end.size() == 1 ? turn_end[0] : LLAMA_TOKEN_NULL;
    std::ifstream in(argv[2]);
    std::ofstream out(argv[4]);
    std::string line;
    int n_done = 0;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        const json c = json::parse(line);
        prompt p;
        if (!build(c, vocab, bricks, h.n_embd, p)) { return 1; }
        if (c.value("check", false)) {
            const json r = check(ctx, h, p, n_vocab);
            std::fprintf(stderr, "bricks check: %s\n", r.dump().c_str());
            out << json({{"check", r}}).dump() << "\n";
            continue;
        }
        fresh(ctx, h);
        h.hide = p.slots;
        h.put = p.put;
        const int P = (int) p.toks.size();
        if (!feed(ctx, h, p.toks, 0, false)) { std::fprintf(stderr, "bricks: decode failed\n"); return 1; }
        h.put.clear();
        const auto first = last_scores(ctx, n_vocab);
        // greedy answer
        std::string said;
        {
            std::vector<float> lg = first;
            for (int k = 0, pos = P; k < MAX_NEW; k++) {
                const llama_token t = (llama_token) (std::max_element(lg.begin(), lg.end()) - lg.begin());
                if (t == stop || llama_vocab_is_eog(vocab, t)) { break; }
                char piece[256];
                const int n = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
                said.append(piece, std::max(n, 0));
                if (!feed(ctx, h, {t}, pos++, false)) { break; }
                lg = last_scores(ctx, n_vocab);
            }
        }
        // every candidate's score: log p of its words after the prompt (as written and capitalized, summed in p)
        json sc = json::object();
        for (const auto & cand : c.value("cands", json::array())) {
            const std::string w = cand.get<std::string>();
            std::vector<std::string> forms = {w};
            std::string up = w;
            if (!up.empty()) { up[0] = (char) std::toupper((unsigned char) up[0]); }
            if (up != w) { forms.push_back(up); }
            std::vector<double> lps;
            for (const auto & f : forms) {
                llama_memory_seq_rm(llama_get_memory(ctx), 0, P, -1);
                const auto t = tokenize(vocab, f);
                double lp = log_softmax_at(first.data(), n_vocab, t[0]);
                if (t.size() > 1) {
                    feed(ctx, h, t, P, true);
                    for (size_t j = 1; j < t.size(); j++) {
                        lp += log_softmax_at(llama_get_logits_ith(ctx, (int) j - 1), n_vocab, t[j]);
                    }
                }
                lps.push_back(lp);
            }
            const double mx = *std::max_element(lps.begin(), lps.end());
            double z = 0;
            for (double x : lps) { z += std::exp(x - mx); }
            sc[w] = mx + std::log(z);
        }
        out << json({{"said", said}, {"scores", sc}, {"slot", p.slots}, {"meta", c.value("meta", json::object())}}).dump()
            << "\n";
        out.flush();
        if (++n_done % 64 == 0) { std::fprintf(stderr, "bricks: %d cases\n", n_done); }
    }
    std::fprintf(stderr, "bricks: %d cases done\n", n_done);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
