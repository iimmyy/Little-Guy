import sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
out = Path(sys.argv[1]); variant = sys.argv[2]; rows = int(sys.argv[3]) if len(sys.argv) > 3 else 128
W = {n: numpy_helper.from_array(np.load(out / f"w_{n}.npy"), f"w_{n}") for n in ("gate", "up", "down")}
nodes = [helper.make_node("MatMul", ["x", "w_gate"], ["g"])]
inits = [W["gate"]]; last = "g"; width = 10240
if variant in ("gateup", "gateup_down", "tanh"):
    nodes.append(helper.make_node("MatMul", ["x", "w_up"], ["u"])); inits.append(W["up"])
    if variant == "tanh":
        nodes.append(helper.make_node("Tanh", ["g"], ["t"])); nodes.append(helper.make_node("Mul", ["t", "u"], ["h"]))
    else:
        nodes.append(helper.make_node("Mul", ["g", "u"], ["h"]))
    last = "h"
    if variant == "gateup_down":
        nodes.append(helper.make_node("MatMul", ["h", "w_down"], ["y"])); inits.append(W["down"]); last = "y"; width = 2560
nodes[-1].output[0] = "y" if last != "y" else nodes[-1].output[0]
g = helper.make_graph(nodes, "ffn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, 2560])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, width])], inits)
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), out / f"v_{variant}.onnx")
print("built", variant)
