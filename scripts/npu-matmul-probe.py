#!/usr/bin/env python3
"""Build and check the gate-2 int4-vs-int8 matmul probe for the Q6A NPU.

`build` writes one float ONNX matmul whose weights sit exactly on a symmetric
per-channel int4 grid, plus QAIRT quantization overrides that store those same
weights as int4 or int8. Both variants therefore hold identical values; only
the stored bit width differs. `compare` checks a board output against the
float reference. Procedure: docs/NPU_PORT.md, "Gate 2 procedure".
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


INPUT_NAME = "x"
WEIGHT_NAME = "w"
OUTPUT_NAME = "y"
INT4_LEVEL = 7  # symmetric grid [-7, 7]; -8 is unused so both int4 and int8 hold it exactly


def on_grid_weights(k: int, n: int, rng: np.random.Generator):
    """Return float32 weights [k, n] exactly representable as scale[n] * q, |q| <= 7."""
    scale = rng.uniform(0.002, 0.02, size=n).astype(np.float32)
    q = rng.integers(-INT4_LEVEL, INT4_LEVEL + 1, size=(k, n)).astype(np.int8)
    # Pin each column's extremes so a min/max quantizer derives the same scale.
    q[0, :] = INT4_LEVEL
    q[1, :] = -INT4_LEVEL
    return (q.astype(np.float32) * scale).astype(np.float32), scale


def weight_encodings(scale: np.ndarray, bitwidth: int):
    """Per-output-channel symmetric encodings in QAIRT's AIMET override format."""
    encodings = []
    for s in scale.tolist():
        limit = s * INT4_LEVEL
        encodings.append({
            "bitwidth": bitwidth,
            "dtype": "int",
            "is_symmetric": "True",
            "min": -limit,
            "max": limit,
            "offset": -(2 ** (bitwidth - 1)),
            "scale": s,
        })
    return encodings


def activation_encoding(values: np.ndarray):
    low, high = float(values.min()), float(values.max())
    return [{"bitwidth": 8, "dtype": "int", "is_symmetric": "False", "min": low, "max": high}]


def overrides(scale, bitwidth, x_all, y_all):
    return {
        "activation_encodings": {
            INPUT_NAME: activation_encoding(x_all),
            OUTPUT_NAME: activation_encoding(y_all),
        },
        "param_encodings": {WEIGHT_NAME: weight_encodings(scale, bitwidth)},
    }


def write_onnx(path: Path, weights: np.ndarray, rows: int):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    k, n = weights.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", [INPUT_NAME, WEIGHT_NAME], [OUTPUT_NAME])],
        path.stem,
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [rows, k])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [rows, n])],
        [numpy_helper.from_array(weights, WEIGHT_NAME)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build(out: Path, k: int, n: int, rows: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    weights, scale = on_grid_weights(k, n, rng)
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    reference = out / "reference"
    inputs.mkdir(exist_ok=True)
    reference.mkdir(exist_ok=True)

    xs, ys, lines = [], [], []
    for index in range(samples):
        x = rng.standard_normal((rows, k)).astype(np.float32)
        y = x @ weights
        x.tofile(inputs / f"x_{index}.raw")
        y.tofile(reference / f"y_{index}.raw")
        xs.append(x)
        ys.append(y)
        lines.append(f"inputs/x_{index}.raw")
    (out / "input_list.txt").write_text("\n".join(lines) + "\n")

    x_all, y_all = np.stack(xs), np.stack(ys)
    for bitwidth in (4, 8):
        (out / f"overrides_int{bitwidth}.json").write_text(
            json.dumps(overrides(scale, bitwidth, x_all, y_all), indent=1)
        )
    write_onnx(out / "matmul.onnx", weights, rows)
    (out / "probe.json").write_text(json.dumps(
        {"k": k, "n": n, "rows": rows, "samples": samples, "seed": seed,
         "weight_bytes_int4": k * n // 2, "weight_bytes_int8": k * n}, indent=1
    ))


def compare(reference: np.ndarray, output: np.ndarray):
    if reference.size != output.size:
        raise ValueError(f"size mismatch: reference {reference.size}, output {output.size}")
    ref = reference.astype(np.float64).ravel()
    out = output.astype(np.float64).ravel()
    denominator = np.linalg.norm(ref) * np.linalg.norm(out)
    cosine = float(ref @ out / denominator) if denominator else 0.0
    return {
        "cosine": cosine,
        "max_abs_error": float(np.max(np.abs(ref - out))),
        "reference_max_abs": float(np.max(np.abs(ref))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="write ONNX, inputs, references, and overrides")
    build_parser.add_argument("out", type=Path)
    build_parser.add_argument("--k", type=int, default=2560, help="input width (E4B hidden size)")
    build_parser.add_argument("--n", type=int, default=10240, help="output width (E4B FFN size)")
    build_parser.add_argument("--rows", type=int, default=1, help="1 = decode step; e.g. 128 = prefill chunk")
    build_parser.add_argument("--samples", type=int, default=8)
    build_parser.add_argument("--seed", type=int, default=6490)

    compare_parser = commands.add_parser("compare", help="check a board output against the float reference")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("output", type=Path)
    compare_parser.add_argument("--min-cosine", type=float, default=0.99)

    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args.out, args.k, args.n, args.rows, args.samples, args.seed)
            print(f"wrote probe to {args.out}")
            return 0
        result = compare(np.fromfile(args.reference, np.float32), np.fromfile(args.output, np.float32))
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    passed = result["cosine"] >= args.min_cosine
    print(json.dumps({**result, "pass": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
