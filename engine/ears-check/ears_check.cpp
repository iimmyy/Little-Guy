// LG: checks his ears in llama.cpp against the PyTorch reference (tools/ears/widen.py), number by number.
//   ears-check feat EARS.gguf SOUND.f32 OUT.f32   the bands the front end makes from raw mono float32 samples at the
//                                                 ears' sample rate (48 kHz for wide ears): [n_mel][n_frames], dims on stdout
//   ears-check embd EARS.gguf BRAIN.gguf SOUND.f32 OUT.f32   what the ears hand his brain: [n_tokens][n_embd]
#include "clip.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-audio.h"

#include <cstdio>
#include <string>
#include <vector>

static std::vector<float> read_f32(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f) / 4;
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> x(n);
    if (std::fread(x.data(), 4, n, f) != (size_t) n) { std::exit(1); }
    std::fclose(f);
    return x;
}

static void write_f32(const char * path, const float * x, size_t n) {
    FILE * f = std::fopen(path, "wb");
    std::fwrite(x, 4, n, f);
    std::fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "see the top of ears_check.cpp\n"); return 1; }
    const std::string mode = argv[1];
    if (mode == "feat") {
        clip_context_params cp{};
        cp.use_gpu = false;
        cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        cp.warmup = false;
        clip_init_result r = clip_init(argv[2], cp);
        if (!r.ctx_a) { std::fprintf(stderr, "no audio in %s\n", argv[2]); return 1; }
        mtmd_audio_preprocessor_gemma4a pre(r.ctx_a);
        pre.initialize();
        const auto x = read_f32(argv[3]);
        std::vector<mtmd_audio_mel> out;
        if (!pre.preprocess(x.data(), x.size(), out) || out.empty()) { std::fprintf(stderr, "preprocess failed\n"); return 1; }
        const auto & m = out[0];
        write_f32(argv[4], m.data.data(), m.data.size());
        std::printf("{\"n_mel\": %lld, \"n_frames\": %lld, \"chunks\": %zu, \"sample_rate\": %d}\n", (long long) m.n_mel,
                    (long long) m.n_len, out.size(), clip_get_hparams(r.ctx_a)->audio_sample_rate);
        return 0;
    }
    if (mode == "embd" && argc >= 6) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[3], mp);
        mtmd_context_params p = mtmd_context_params_default();
        p.use_gpu = false;
        p.print_timings = false;
        p.warmup = false;
        mtmd_context * ctx = mtmd_init_from_file(argv[2], model, p);
        if (!ctx) { return 1; }
        const auto x = read_f32(argv[4]);
        mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(x.size(), x.data());
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text txt{mtmd_default_marker(), false, true};
        const mtmd_bitmap * bmps[1] = {bmp};
        if (mtmd_tokenize(ctx, chunks, &txt, bmps, 1) != 0) { return 1; }
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_AUDIO) { continue; }
            if (mtmd_encode_chunk(ctx, c) != 0) { return 1; }
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(c);
            const int n_embd = llama_model_n_embd_inp(model);
            write_f32(argv[5], mtmd_get_output_embd(ctx), n_tok * n_embd);
            std::printf("{\"n_tokens\": %zu, \"n_embd\": %d}\n", n_tok, n_embd);
            return 0;
        }
        return 1;
    }
    std::fprintf(stderr, "unknown mode\n");
    return 1;
}
