#!/usr/bin/env python3
"""Little Guy's pixel faces (Immy, 2026-10-25: pixel-art robot faces, glowing on black, "cuteness with coolness").

Faces are drawn by hand on a 32 x 18 grid: the same 16:9 shape as his screen (EDO E555HBM2, lit area 122.9 x 69.1 mm),
so each face pixel is a square ~3.8 mm wide on his face. Lines are one pixel thick, pixels solid with no gaps.

usage: faces.py sheet <out.png>            all faces side by side
       faces.py texture <name> <out.png>   one face as a screen picture for the simulator
"""
import sys

from PIL import Image, ImageDraw, ImageFont

GX, GY = 32, 18
CYAN, RED, BLACK = (70, 215, 245), (235, 45, 45), (6, 6, 8)


def line(a, b):
    """1-pixel line between two grid points"""
    im = Image.new("1", (GX, GY))
    ImageDraw.Draw(im).line([a, b], fill=1, width=1)
    return [(x, y) for x in range(GX) for y in range(GY) if im.getpixel((x, y))]


def box(x0, y0, x1, y1):
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def both(pts):
    """a left-side feature and its mirror image on the right"""
    return pts + [(GX - 1 - x, y) for x, y in pts]


ARC = [(7, 8), (8, 7), (9, 6), (10, 6), (11, 7), (12, 8)]
EYE = box(9, 6, 10, 8)
SMILE = [(12, 12), (13, 13), (14, 14), (15, 14), (16, 14), (17, 14), (18, 13), (19, 12)]
FROWN = [(12, 14), (13, 13), (14, 12), (15, 12), (16, 12), (17, 12), (18, 13), (19, 14)]
SHADES_GLINT = [(7, 8), (8, 7), (19, 8), (20, 7)]

FACES = {   # name: (lit pixels, colour, dark pixels drawn on top)
    "happy": (both(ARC) + SMILE, CYAN, []),
    "cute": (both(EYE) + [(13, 12), (14, 13), (15, 12), (16, 12), (17, 13), (18, 12)], CYAN, []),
    "wink": (ARC + [(GX - 1 - x, y) for x, y in EYE] + SMILE, CYAN, []),
    "shades": (line((5, 6), (26, 6)) + box(6, 7, 13, 9) + box(18, 7, 25, 9) + [(15, 13), (16, 13), (17, 13), (18, 13), (19, 12)],
               CYAN, SHADES_GLINT),
    "smirk": (both(line((8, 7), (11, 7)) + box(9, 8, 10, 8)) + [(13, 13), (14, 13), (15, 13), (16, 13), (17, 13), (18, 12)]
              + line((19, 5), (22, 4)), CYAN, []),
    "sad": (both(line((7, 8), (11, 8))) + [(8, 10)] + FROWN, CYAN, []),
    "angry": (both(line((7, 5), (11, 7)) + box(10, 8, 11, 9)) + FROWN, RED, []),
    "dizzy": (both(line((7, 5), (11, 9)) + line((7, 9), (11, 5))) + [(12 + i, 13 - i % 2) for i in range(8)], CYAN, []),
}


def draw(name, cell):
    lit, colour, dark = FACES[name]
    img = Image.new("RGB", (GX * cell, GY * cell), BLACK)
    d = ImageDraw.Draw(img)
    for pts, c in ((set(lit), colour), (dark, BLACK)):
        for x, y in pts:
            d.rectangle([x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1], fill=c)
    return img


def texture(name, path):
    draw(name, 40).transpose(Image.FLIP_TOP_BOTTOM).save(path)     # MuJoCo reads textures bottom row first


def sheet(path, cell=12, cols=4):
    font = ImageFont.load_default(size=20)
    tw, th = GX * cell, GY * cell
    rows = (len(FACES) + cols - 1) // cols
    img = Image.new("RGB", (cols * (tw + 20) + 20, rows * (th + 50) + 50), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((20, 12), f"{GX} x {GY} pixels, solid, 1-pixel lines (each pixel ~3.8 mm on his face)", fill=(20, 20, 20), font=font)
    for i, name in enumerate(FACES):
        x0, y0 = 20 + (i % cols) * (tw + 20), 45 + (i // cols) * (th + 50)
        img.paste(draw(name, cell), (x0, y0))
        d.text((x0 + tw // 2, y0 + th + 18), name, fill=(60, 60, 60), font=font, anchor="mm")
    img.save(path)


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[2])
    else:
        texture(sys.argv[2], sys.argv[3])
    print("wrote", sys.argv[-1])
