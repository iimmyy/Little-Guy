import sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
out = Path(sys.argv[1]); variant = sys.argv[2]; rows = int(sys.argv[3]) if len(sys.argv) > 3 else 128
W = {n: numpy_helper.from_array(np.load(out / f"w_{n}.npy"), f"w_{n}") for n in ("gate", "up", "down")}
nodes = [helper.make_node("MatMul", ["x", "w_gate"], ["g"])]
inits = [W["gate"]]; last = "g"; width = 10240
if variant in ("gateup", "gateup_down", "tanh", "split_down", "full_split"):
    nodes.append(helper.make_node("MatMul", ["x", "w_up"], ["u"])); inits.append(W["up"])
    if variant == "full_split":  # the real FFN: gelu_tanh(gate) * up, with the built-in Gelu op
        nodes.append(helper.make_node("Gelu", ["g"], ["ga"], approximate="tanh"))
        nodes.append(helper.make_node("Mul", ["ga", "u"], ["h"]))
    elif variant == "tanh":
        nodes.append(helper.make_node("Tanh", ["g"], ["t"])); nodes.append(helper.make_node("Mul", ["t", "u"], ["h"]))
    else:
        nodes.append(helper.make_node("Mul", ["g", "u"], ["h"]))
    last = "h"
    if variant == "gateup_down":
        nodes.append(helper.make_node("MatMul", ["h", "w_down"], ["y"])); inits.append(W["down"]); last = "y"; width = 2560
    if variant in ("split_down", "full_split"):  # down's 10240-long sums as 4 sums of 2560, the shape gate and up run fast at
        wd = np.load(out / "w_down.npy"); parts = []
        for i in range(4):
            s = slice(i * 2560, (i + 1) * 2560)
            starts = numpy_helper.from_array(np.array([i * 2560], np.int64), f"s{i}")
            ends = numpy_helper.from_array(np.array([(i + 1) * 2560], np.int64), f"e{i}")
            axes = numpy_helper.from_array(np.array([1], np.int64), f"a{i}")
            inits += [starts, ends, axes, numpy_helper.from_array(np.ascontiguousarray(wd[s]), f"w_down{i}")]
            nodes.append(helper.make_node("Slice", ["h", f"s{i}", f"e{i}", f"a{i}"], [f"h{i}"]))
            nodes.append(helper.make_node("MatMul", [f"h{i}", f"w_down{i}"], [f"y{i}"]))
            parts.append(f"y{i}")
        nodes.append(helper.make_node("Add", [parts[0], parts[1]], ["y01"]))
        nodes.append(helper.make_node("Add", [parts[2], parts[3]], ["y23"]))
        nodes.append(helper.make_node("Add", ["y01", "y23"], ["y"]))
        last = "y"; width = 2560
nodes[-1].output[0] = "y" if last != "y" else nodes[-1].output[0]
g = helper.make_graph(nodes, "ffn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, 2560])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, width])], inits)
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 20 if variant == "full_split" else 13)]), out / f"v_{variant}.onnx")
print("built", variant)
