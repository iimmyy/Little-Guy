# Getting Gemma 4 onto the Q6A's NPU (plan, WIP)

Here's the goal. I want Gemma 4 running on the Q6A's NPU, which is a Hexagon v68, with the audio, vision, and language parts all on board. E2B first, and E4B is the one I really want. Qualcomm's official LLM runtime, Genie, says you need a Hexagon v73 or newer, and as far as I can tell nobody's gotten a 2B sized model running on a v68. The only v68 LLMs I've seen are around 1B, Radxa's Llama 3.2 1B and a community Qwen3.5 0.8B build, both on Ubuntu. So yeah, this is the challenge, and that's kind of the whole point.

Android was just my test setup. I've moved this work over to Ubuntu on the Q6A because that's where all the NPU stuff (`fastrpc`, `libcdsprpc`, the QAIRT runtime) is documented, and honestly Android eating all my RAM is what killed my E4B attempt in the first place. The switch is written up in [Q6A_LINUX_BOOT.md](Q6A_LINUX_BOOT.md).

## What's actually inside the model

These numbers come from the [Gemma 4 technical report](https://arxiv.org/abs/2607.02770), Table 1:

| Part | E2B | E4B | Where I think it'll run |
| --- | --- | --- | --- |
| Audio encoder (takes 40 ms chunks of sound) | 305M | 305M | NPU |
| Vision encoder | 150M | 150M | NPU |
| Token embedder, which doubles as the output layer (262k word vocab) | 400M | 670M | NPU or GPU, I'll have to measure |
| Per-layer embeddings (PLE) | 2,340M | 2,820M | regular RAM on the CPU side, since it only reads the rows for the current token |
| The actual transformer layers | 1,870M | 3,940M | NPU |
| MTP drafter, a little helper that guesses tokens ahead | 76M | 77M | NPU |

Some other stuff that matters for this:

- E2B and E4B each have their own separate audio and vision encoders. Only the 12B model got rid of them. That means I can compile the encoders on their own.
- The attention layers go 4 local ones (they only look at nearby tokens) to every 1 global one. Some layers also share their memory of past tokens, 20 of 35 layers in E2B and 18 of 42 in E4B. Whatever I compile has to deal with both of those.
- Google already trained special "mobile" versions that are squished down to 2 and 4 bit weights. The text part comes out to about 0.8 GB for E2B and 2.3 GB for E4B.
- The report never mentions MatFormer for Gemma 4, which was the trick in the older Gemma 3n where a smaller model was nested inside the bigger one. So I'm not counting on being able to cut a smaller model out of E4B.

## The speed math

The Q6A's memory tops out at about 22 GB/s. Every time the model writes one token, it basically has to read all of its transformer weights plus the output layer out of memory once. So how fast it can write mostly comes down to how many bytes it has to read. These are best case guesses. The "realistic" column assumes I only get about 60% of the max, which is roughly what Llama 1B gets on this board.

| Model and weight size | Read per token | Best case at 22 GB/s | Realistic (~60%) |
| --- | --- | --- | --- |
| E2B, 8 bit | ~2.3 GB | ~9.7 tok/s | ~5.8 tok/s |
| E2B, 4 bit | ~1.1 GB | ~19 tok/s | ~11.6 tok/s |
| E4B, 8 bit | ~4.6 GB | ~4.8 tok/s | ~2.9 tok/s |
| E4B, 4 bit | ~2.3 GB | ~9.5 tok/s | ~5.7 tok/s |

Right now my E2B on the Android GPU writes at about 6 to 7 tok/s.

There are three ways I can beat those numbers:

1. **4 bit weights.** Half the bytes means double the speed ceiling. The catch is that Qualcomm's main 4 bit format (they call it LPBQ) needs a v69 or newer, so I can't use it. Google's mobile versions use a simpler per-channel style of 4 bit though, and whether the v68 can handle that is the biggest unknown in this whole plan.
2. **Guessing ahead with the drafter.** Since the slow part is reading the weights, checking a few guessed tokens at once costs about the same as writing one. Every guess that turns out right is basically free speed. My Android logs already show LiteRT running a "verify" step, so I think my E2B on Android already does this.
3. **Keeping the camera and mic work out of the way.** I don't want the vision and audio encoders fighting the language part for the same hardware.

## The checkpoints

I'm doing this in order, and I don't move on until the current one works.

0. **Does the NPU even work on Ubuntu? Yes! Passed on 2026-09-28.** Radxa's prebuilt Llama 3.2 1B ran on the NPU. It read an 823 token prompt at about 400 tok/s and wrote at 9 to 10 tok/s. The whole model is one "weight sharing" file, which I thought needed a newer chip, and Genie can take embeddings directly with `-e`, which I'll need for the vision and audio stuff. All the details are in [Q6A_LINUX_BOOT.md](Q6A_LINUX_BOOT.md).
1. **Get a machine that can compile.** Qualcomm's compile tools really want an x86 Ubuntu machine. On ARM Linux they can only convert ONNX files, and any VM on my Mac is ARM. So I need an x86 Ubuntu VM or a cloud box, with a QAIRT version that matches what's on the board.
2. **Does the v68 handle 4 bit?** I build one single matrix multiply as 4 bit and again as 8 bit, run both on the board, and check that the output is right and how fast each one is. This answers whether E4B ends up around 3 tok/s or around 6.
3. **One layer of E2B.** I pull one local attention layer and one global attention layer out of the Hugging Face weights, export them to ONNX with fixed sizes (one version for writing a single token, one for reading a chunk of prompt), compile them for v68, run them on the board, and compare against PyTorch. I want the outputs to match with a cosine similarity of 0.99 or better. I'll also time each layer.
4. **All of E2B's language part.** I split the model into a few files with a few layers each, find out how much memory the NPU will let me use, and keep the PLE lookup, the tokenizer, and picking the next word on the CPU. Once the numbers check out, I run my test prompts from [MODEL_SELECTION.md](MODEL_SELECTION.md) against my GPU E2B to compare.
5. **The encoders and the drafter.** Compile the vision encoder, audio encoder, and drafter as their own pieces, and test them one at a time through the real setup.
6. **E4B.** Same recipe, bigger model. I'll only squish it further or distill it (on cloud GPUs) as much as the numbers tell me I have to.

## How to run checkpoint 2

The test is one matrix multiply that's the same shape as one of E4B's feed-forward layers, 2560 in and 10240 out, about 26 million weights. `scripts/npu-matmul-probe.py` makes weights that fit perfectly into 4 bits, and then I save those exact same weights once as 4 bit and once as 8 bit. That way the only thing different between the two builds is the bit size. The compile steps follow [Radxa's QAIRT example](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage), which uses QAIRT 2.42 on an x86 Linux machine. I haven't actually run these QAIRT commands yet, so I'll double check each flag with `--help` on whatever version I install.

On the x86 machine:

```sh
python3 scripts/npu-matmul-probe.py build .run/probe            # one token shape, about 110 MB, Git ignores it
python3 scripts/npu-matmul-probe.py build .run/probe128 --rows 128   # prompt chunk shape
cd .run/probe
for bits in 4 8; do
  qairt-converter --input_network matmul.onnx -d x 1,2560 \
    --quantization_overrides overrides_int${bits}.json -o matmul_int${bits}.dlc
  qairt-quantizer --input_dlc matmul_int${bits}.dlc --input_list input_list.txt \
    --weights_bitwidth ${bits} --act_bitwidth 8 --use_per_channel_quantization \
    --output_dlc matmul_int${bits}_q.dlc
  qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHtp.so \
    --dlc_path matmul_int${bits}_q.dlc --config_file config_file.json \
    --output_dir out --binary_file matmul_int${bits}
done
```

The `config_file.json` and `config_backend.json` files come from Radxa's example, set to `"dsp_arch": "v68"` and `"soc_id": 35`. The `graph_names` in there has to match the graph name inside the DLC file.

Then on the Q6A, I copy over `qnn-net-run`, `libQnnHtp.so`, `libQnnHtpV68Stub.so`, `libQnnHtpV68Skel.so`, both `.bin` files, the `inputs/` folder, and `input_list.txt`, and run:

```sh
./qnn-net-run --backend ./libQnnHtp.so --retrieve_context ./matmul_int4.bin \
  --input_list ./input_list.txt --output_dir out4 --profiling_level basic
python3 npu-matmul-probe.py compare reference/y_0.raw out4/Result_0/y.raw
```

Then same thing with the 8 bit build. If it complains it can't find the skel library, I point `ADSP_LIBRARY_PATH` at the folder it's in. For timing, I'll repeat the input lines enough that each run takes a few seconds, then read the results with `qnn-profile-viewer`.

How I'll read what comes out:

- **First, check the file sizes.** The 4 bit `.bin` should be about half the size of the 8 bit one, roughly 13 MB of weights against 26. If they're the same size, the tools quietly saved it as 8 bit anyway, and that tells me everything I need to know about 4 bit on this chip.
- **If 4 bit compiles, the output matches (0.99 or better), and it's clearly faster:** 4 bit works, and E4B is looking like about 6 tok/s before the drafter helps.
- **If 4 bit works but runs the same speed as 8 bit:** the chip is probably unpacking the weights before using them, so I get no speed out of it. Then I plan around 8 bit.
- **If it won't compile for v68, or the output's wrong:** 8 bit it is, and for E4B I lean on distilling and the drafter.

The 128 row version tells me whether reading prompts is limited by raw compute. It doesn't change the 4 bit answer either way.

## What I write down every time

The model and exactly which version it came from, with hashes. The QAIRT and QNN versions. The shapes and context length. How each part was quantized. How close the output matched PyTorch. Time to first word, reading and writing speed, peak memory, temperature, power, and whether the fan was on. Getting something to compile doesn't count. It only counts when it gives the right answer, on the board, and I've measured it.

References: [Gemma 4 technical report](https://arxiv.org/abs/2607.02770), [Genie LLM tutorial (v73+ requirement)](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie), [v68 Llama export issue](https://github.com/qualcomm/ai-hub-models/issues/337), [Radxa Llama 3.2 1B v68](https://docs.radxa.com/dragon/q6a/app-dev/npu-dev/llama3.2-1b-qairt-v68), [Qwen3.5-0.8B on Q6A NPU](https://huggingface.co/PrismPhi/Qwen3.5-0.8B-Radxa-Dragon-Q6A-QCS6490-QNN-NPU), [QAIRT int4 weight encodings](https://docs.qualcomm.com/nav/home/htp_guidelines_int4_weights.html?product=1601111740010412), [LPBQ requires v69+](https://quic.github.io/aimet-pages/releases/latest/techniques/lpbq.html), [QAIRT Linux host setup](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html).
