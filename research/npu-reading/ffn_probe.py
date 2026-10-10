"""Step 1 of NPU reading: one real E4B FFN (layer 10) as an 8-bit NPU program, timed and checked.

  export (Mac, needs safetensors + numpy):  python ffn_probe.py export OUT_DIR [ROWS]
      pulls layer 10's gate/up/down weights from the master brain, writes random calibration inputs
      shaped like the FFN's normalized input, and the float32 reference outputs
  build (QAIRT VM, needs onnx + numpy):      python ffn_probe.py build OUT_DIR [ROWS]
      writes ffn.onnx: y = down(gelu_tanh(x @ gate) * (x @ up)), with x as ROWS words of 2560 numbers
  compare (anywhere with numpy):             python ffn_probe.py compare REFERENCE.raw RESULT.raw
"""
import sys
from pathlib import Path

import numpy as np

LAYER, HIDDEN, FF = 10, 2560, 10240
MASTER = Path(__file__).resolve().parents[2] / ".run/weights/masters/e4b-qat/model.safetensors"


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def export(out, rows):
    from safetensors import safe_open
    out.mkdir(parents=True, exist_ok=True)
    pre = f"model.language_model.layers.{LAYER}.mlp."
    with safe_open(MASTER, "pt") as f:
        w = {n: f.get_tensor(pre + n + "_proj.weight").float().numpy() for n in ("gate", "up", "down")}
    np.save(out / "w_gate.npy", w["gate"].T.copy())   # [2560, 10240] so the graph is x @ w
    np.save(out / "w_up.npy", w["up"].T.copy())
    np.save(out / "w_down.npy", w["down"].T.copy())   # [10240, 2560]
    rng = np.random.default_rng(0)
    (out / "inputs").mkdir(exist_ok=True)
    (out / "reference").mkdir(exist_ok=True)
    for i in range(4):  # calibration and test inputs: RMS-normalized words, like the FFN really receives
        x = rng.standard_normal((rows, HIDDEN)).astype(np.float32)
        x /= np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        y = (gelu_tanh(x @ np.load(out / "w_gate.npy")) * (x @ np.load(out / "w_up.npy"))) @ np.load(out / "w_down.npy")
        x.tofile(out / "inputs" / f"x_{i}.raw")
        y.astype(np.float32).tofile(out / "reference" / f"y_{i}.raw")
    print(f"exported layer {LAYER} FFN and 4 inputs of {rows} words to {out}")


def build(out, rows):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    inits = [numpy_helper.from_array(np.load(out / f"w_{n}.npy"), f"w_{n}") for n in ("gate", "up", "down")]
    c = lambda name, v: numpy_helper.from_array(np.array(v, dtype=np.float32), name)
    inits += [c("c_half", 0.5), c("c_one", 1.0), c("c_k", 0.7978845608), c("c_a", 0.044715)]
    nodes = [
        helper.make_node("MatMul", ["x", "w_gate"], ["g"]),
        helper.make_node("MatMul", ["x", "w_up"], ["u"]),
        helper.make_node("Mul", ["g", "g"], ["g2"]),
        helper.make_node("Mul", ["g2", "g"], ["g3"]),
        helper.make_node("Mul", ["g3", "c_a"], ["g3a"]),
        helper.make_node("Add", ["g", "g3a"], ["inner"]),
        helper.make_node("Mul", ["inner", "c_k"], ["inner_k"]),
        helper.make_node("Tanh", ["inner_k"], ["t"]),
        helper.make_node("Add", ["t", "c_one"], ["t1"]),
        helper.make_node("Mul", ["g", "t1"], ["gt"]),
        helper.make_node("Mul", ["gt", "c_half"], ["gelu"]),
        helper.make_node("Mul", ["gelu", "u"], ["h"]),
        helper.make_node("MatMul", ["h", "w_down"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "ffn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, HIDDEN])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, HIDDEN])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, out / "ffn.onnx")
    (out / "input_list.txt").write_text("".join(f"x:={out / 'inputs' / f'x_{i}.raw'}\n" for i in range(4)))
    print(f"built {out / 'ffn.onnx'} ({(out / 'ffn.onnx').stat().st_size / 1e6:.0f} MB)")


def compare(ref, got):
    a, b = np.fromfile(ref, np.float32), np.fromfile(got, np.float32)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    print(f"cosine {cos:.5f}   relative error {rel:.4f}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "compare":
        compare(*args)
    else:
        rows = int(args[1]) if len(args) > 1 else 128
        {"export": export, "build": build}[cmd](Path(args[0]).resolve(), rows)
