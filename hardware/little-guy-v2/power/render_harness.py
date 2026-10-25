"""Render a logical harness overview; no connector-view pinout implied."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent


def main():
    model = json.loads((ROOT / "harness.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 11), facecolor="#faf9f6")
    ax.set(xlim=(0, 12), ylim=(0, 11))
    ax.axis("off")
    red, blue, green, ink = "#b54d48", "#376d9a", "#638359", "#29333d"

    def box(x, y, w, h, text, face="#ffffff", size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.05",
                                   facecolor=face, edgecolor="#87929c", linewidth=.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=ink)

    ax.text(.2, 10.65, "One power feed per servo", fontsize=22, weight="bold", color=ink)
    ax.text(.2, 10.27, "Logical candidate • ratings and physical pin orientation not assigned", fontsize=11, color=ink)
    box(.3, 8.8, 2.4, .75, "Jetson on stand\nOwn specified supply")
    box(3.1, 8.8, 2.5, .75, "USB adapter\nUSB logic power: verify")
    ax.plot([2.75, 3.05], [9.17, 9.17], color=ink, linewidth=2)
    ax.text(2.9, 9.45, "USB", ha="center", fontsize=9)
    ax.text(6.0, 9.4, "Adapter servo V+ omitted", fontsize=11, color=red)
    ax.text(6.0, 9.02, "DATA can stay powered by USB", fontsize=10, color=green)
    ax.plot([5.6, 11.15, 11.15], [8.92, 8.92, 1.32], color=green, linewidth=1.6)
    ax.text(11.4, 5.3, "Shared DATA membership • physical routing pending", rotation=90,
            ha="center", va="center", fontsize=10, color=green)

    box(.3, 5.6, 2.35, 1.15, "Servo supply\nNominal 12 V\nModel unselected")
    box(.3, 4.0, 2.35, 1.05, "Source protection\n+ manual disconnect\nParts unselected", size=9)
    ax.plot([1.45, 1.45], [5.55, 5.1], color=red, linewidth=2)
    ax.plot([1.45, 1.45, 4.1, 4.1], [3.95, 3.65, 3.65, 8.05], color=red, linewidth=2)
    ax.plot([2.7, 3.5, 3.5, 4.6], [6.17, 6.17, 3.15, 3.15], color=blue, linewidth=2)
    ax.plot([4.6, 4.6], [1.2, 8.05], color=blue, linewidth=2)
    ax.plot([4.1, 4.1], [1.2, 3.65], color=red, linewidth=2)
    ax.plot(4.1, 3.65, "o", color=red, markersize=3)
    ax.plot(4.6, 3.15, "o", color=blue, markersize=3)
    ax.text(3.95, 8.05, "Hub", ha="right", fontsize=12, color=ink)
    ax.text(.3, 2.3, "Main trunk reference sums\n9.9 A rated / 29.7 A stalled\n\nNot measured demand\nor fuse/supply selections", fontsize=10, color=ink, linespacing=1.5)
    # Deliberate reference bond, not an additional motor-return feed.
    ax.plot([3.35, 3.35, 4.6], [8.75, 8.42, 8.42], color=blue, linestyle="--", linewidth=1.3)
    ax.plot([4.6, 4.6], [8.42, 8.05], color=blue, linestyle="--", linewidth=1.3)
    ax.text(1.95, 7.65, "One reference bond\nCheck all external\nground paths", fontsize=9, color=blue)

    for i, load in enumerate(model["loads"]):
        y = 7.8 - i * .6
        box(7.3, y - .23, 3.1, .43, load["name"].replace("_", " "), size=10)
        ax.plot([4.1, 5.1], [y + .07, y + .07], color=red, linewidth=1.3)
        ax.add_patch(Rectangle((5.1, y - .01), .45, .16, facecolor="#fff8e6", edgecolor=red, linewidth=1))
        ax.plot([5.55, 7.25], [y + .07, y + .07], color=red, linewidth=1.3)
        ax.plot([4.6, 7.25], [y - .12, y - .12], color=blue, linewidth=1.3)
        ax.plot([10.45, 11.15], [y, y], color=green, linewidth=1)
        for x, join_y, color in ((4.1, y + .07, red), (4.6, y - .12, blue), (11.15, y, green)):
            ax.plot(x, join_y, "o", color=color, markersize=3)
    ax.text(5.33, 8.24, "Individual\nprotection", ha="center", fontsize=9, color=red)
    ax.text(6.32, 1.0, "Each leaf: 0.9 A rated / 2.7 A stall reference", ha="center", fontsize=10, color=ink)
    ax.text(.2, .68, "Red: positive supply     Blue: return/reference     Green: DATA     Crossings without dots do not connect", fontsize=9, color=ink)
    ax.text(.2, .35, "Connectivity checked only. No physical harness built; no electrical or host-loss validation.", fontsize=10, color=ink)
    fig.subplots_adjust(left=.025, right=.985, top=.99, bottom=.015)
    fig.savefig(ROOT / "harness-overview.svg")
    fig.savefig(ROOT / "harness-overview.png", dpi=160)


if __name__ == "__main__":
    main()
