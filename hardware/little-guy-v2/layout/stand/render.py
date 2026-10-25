"""Plot stand sensitivity; the support rectangles are candidate contact spans."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from stability import G

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / "results.json").read_text())
    parameters = json.loads((ROOT / "parameters.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor="#faf9f6")
    colors = ["#7b8288", "#bf7944", "#467aa7", "#3f8e7d"]
    forces = np.linspace(0, 10, 101)
    height = parameters["force_application_height_above_support_m"]
    for row, color in zip(report["support_sensitivity"], colors):
        support = row["support"]
        center = np.array(support["center_xy_mm"])
        size = np.array(support["size_xy_mm"])
        style = "--" if support["name"] == "platform_size_comparison" else "-"
        axes[0].add_patch(Rectangle(center - size / 2, *size, fill=False, edgecolor=color,
                                    linewidth=1.7, linestyle=style))
        zero = next(case for case in row["force_cases"] if case["horizontal_force_n"] == 0)
        needed = np.max([np.maximum(0, (forces * height - edge["gravity_restoring_moment_lower_bound_nm"])
                                    / (G * edge["added_centered_mass_lever_m"])) for edge in zero["edges"]], axis=0)
        axes[1].plot(forces, needed, color=color, linestyle=style, linewidth=2, label=support["label"])
    com_center = np.array(report["jetson_fixed_horizontal_first_moment_kg_m"]) / report["modeled_mass_kg"] * 1000
    radius = report["analytical_arm_bound"]["two_arm_horizontal_first_moment_bound_kg_m"] / report["modeled_mass_kg"] * 1000
    axes[0].add_patch(Circle(com_center, radius, facecolor="#b4c8b4", alpha=.35, edgecolor="#668468"))
    axes[0].plot(0, 0, "+", color="#29333d", markersize=10)
    axes[0].plot(0, -100, "s", color="#bf7944", markersize=5)
    axes[0].annotate("Jetson kit centre", (0, -100), (-175, -215), fontsize=9,
                     arrowprops=dict(arrowstyle="-", color="#6b7278"))
    axes[0].text(0, 8, "Model COM bound\n1.14 kg partial model", ha="center", fontsize=10)
    axes[0].set(xlim=(-220, 220), ylim=(-235, 220), xlabel="Left / right, mm", ylabel="Front / rear, mm",
                title="Candidate support contacts • top view")
    axes[0].set_aspect("equal")
    axes[1].set(xlim=(0, 10), ylim=(0, 7), xlabel="Example outward horizontal force, N",
                ylabel="Added mass at support centre, kg",
                title="Mass needed to reach the model's tipping boundary")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].axvline(5, color="#aaa69e", linewidth=.8, linestyle=":")
    for ax in axes:
        ax.set_facecolor("#faf9f6")
        ax.grid(alpha=.12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stand footprint and weight • static sensitivity", x=.06, ha="left", fontsize=20)
    fig.text(.06, .9, "Force height: 0.55 m. Conservative arm-moment bound; Jetson stays on the stand.", fontsize=10)
    fig.text(.06, .06, "Thresholds have zero added design margin. Unknown masses, sliding, dynamics and structural strength are not validated.", fontsize=9)
    fig.text(.06, .025, "Dashed rectangle matches the assembly shelf only as a comparison; it is not an existing floor base.", fontsize=9, color="#6b7278")
    fig.subplots_adjust(left=.08, right=.97, top=.8, bottom=.19, wspace=.32)
    fig.savefig(ROOT / "stand-sensitivity.png", dpi=160)
    fig.savefig(ROOT / "stand-sensitivity.svg")


if __name__ == "__main__":
    main()
