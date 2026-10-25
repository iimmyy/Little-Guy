#!/usr/bin/env python3
"""Every real part in Little Guy V2 with its weight and where that weight comes from (Immy, 2026-10-25: "we have to put
all our parts into him ... so he weighs correctly").

Each part: (grams, source, body it rides in). Sources: "datasheet" / "maker spec" = the maker's own number,
"listing" = the shop page, "estimate" = my guess until the part is in hand and weighed. Estimates get replaced by
kitchen-scale numbers as parts arrive.

usage: parts.py   prints the list with totals per body and how much is still estimated
"""

PARTS = {
    # head
    "head shell, 2 cameras, mic board, screws": (115, "estimate (was 160 g with a 45 g 3.5\" screen)", "head"),
    "AMOLED panel EDO E555HBM2 5.55\"": (26, "datasheet (EverDisplay spec v05: 26 g +-10%)", "head"),
    "AMOLED HDMI driver board": (25, "estimate (AliExpress combo listing gives no weight)", "head"),
    "neck nod servo ST3215": (55, "maker spec (Feetech, 12 V version)", "head"),
    "neck second servo ST3215 (nod + tilt shared by two)": (55, "maker spec (Feetech, 12 V version)", "head"),
    "neck ball joint + 2 push rods": (15, "estimate (steel ball, two short rods with ball ends)", "head"),
    "ear speakers, 2x 40 mm full-range driver + back cups": (70, "estimate (driver not chosen; ~30 g driver + ~5 g cup)",
                                                            "head"),
    "rubber edge round each ear (2), TPU": (4, "estimate (2 mm round ring, ~1.8 cm3 of TPU each)", "head"),
    # torso (extras on top of ToddlerBot's scaled torso weight)
    "carry handles (2), printed": (30, "estimate (PETG-CF, ~15 g each)", "torso"),
    "woofer, 2\" long-throw, in its sealed chamber": (100, "estimate (driver not chosen; ~60 g driver + ~40 g chamber)",
                                                      "torso"),
}


def grams(body):
    return sum(g for g, _, b in PARTS.values() if b == body)


def main():
    bodies = sorted({b for _, _, b in PARTS.values()})
    for body in bodies:
        print(f"{body}: {grams(body)} g")
        for name, (g, src, b) in PARTS.items():
            if b == body:
                print(f"  {g:5d} g  {name}  [{src}]")
    est = sum(g for g, src, _ in PARTS.values() if src.startswith("estimate"))
    print(f"total {sum(g for g, _, _ in PARTS.values())} g, of which {est} g still estimated")


if __name__ == "__main__":
    main()
