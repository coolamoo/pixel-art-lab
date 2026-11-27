#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无视觉反馈时的结构预览：把像素图缩成字符画，用来看轮廓和色块分布。

用途：当多模态看图能力不可用（接口报错、离线、配额用尽）时，
至少能确认"形状有没有、比例对不对、有没有大片空洞"。

局限：看不出配色美不美、线条干不干净。只做形状兜底。

用法：
    python3 ascii_preview.py 图.png            # 默认 60 列
    python3 ascii_preview.py 图.png -w 80
    python3 ascii_preview.py 图.png -w 80 --mode alpha   # 只看不透明区域
    python3 ascii_preview.py 图.png -w 80 --mode color   # 按色相分类
"""
import sys
import argparse

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow: pip install pillow")


def cls_color(c, bg):
    r, g, b = c[:3]
    if abs(r - bg[0]) < 12 and abs(g - bg[1]) < 12 and abs(b - bg[2]) < 12:
        return "."
    v = (r + g + b) / 3
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 16:                       # 灰阶
        return "@" if v < 60 else ("+" if v < 130 else ("o" if v < 200 else " "))
    if r > g and r > b:
        return "#" if v < 170 else "c"     # 红/橙系
    if g >= r and g >= b:
        return "g"                          # 绿系
    return "b"                              # 蓝/紫系


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-w", "--width", type=int, default=60)
    ap.add_argument("--mode", choices=["alpha", "color", "luma"], default="color")
    ap.add_argument("--bg", default="20,28,36", help="背景色 R,G,B")
    a = ap.parse_args()

    im = Image.open(a.image).convert("RGBA")
    ratio = im.height / im.width
    w = a.width
    h = max(1, int(w * ratio * 0.5))       # 字符高宽比约 2:1
    small = im.resize((w, h), Image.LANCZOS)
    bg = tuple(int(x) for x in a.bg.split(","))

    for y in range(h):
        line = []
        for x in range(w):
            px = small.getpixel((x, y))
            if a.mode == "alpha":
                line.append("#" if px[3] > 90 else ".")
            elif a.mode == "luma":
                if px[3] < 40:
                    line.append(" ")
                else:
                    v = (px[0] * 299 + px[1] * 587 + px[2] * 114) // 1000
                    line.append(" .:-=+*#%@"[min(9, v * 10 // 256)])
            else:
                if px[3] < 40:
                    line.append(" ")
                else:
                    line.append(cls_color(px, bg))
        print("".join(line))


if __name__ == "__main__":
    main()
