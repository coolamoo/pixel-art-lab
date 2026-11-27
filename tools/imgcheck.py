#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""imgcheck —— 像素图/设定图自检（纯 OpenCV + Pillow，无模型依赖）

用法：
  python3 tools/imgcheck.py 图.png [--grid 12] [--ref 参考.png] [--json]

检查项：
  1 尺寸 / 模式 / 是否带 alpha
  2 内容 bbox、四边留白、是否贴边（出框）
  3 连通域：数量、各块面积与 bbox（可发现"缺肢体 / 断成几截"）
  4 调色板：色数、主色占比（可发现"糊成一片 / 颜色串了"）
  5 若给 --ref：与参考图的不透明区域覆盖率 & 尺寸比
  6 若给 --grid：按网格量化，报告格内是否混色（判断像素粒度是否统一）
"""
import sys, json
import numpy as np
import cv2


def load(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit("读不到图: %s" % path)
    if im.ndim == 3 and im.shape[2] == 4:
        bgr, a = im[:, :, :3], im[:, :, 3]
    else:
        bgr = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        a = np.full(bgr.shape[:2], 255, np.uint8)
    return bgr, a


def alpha_mask(a, thr=110, bgr=None):
    """有透明通道就用它；整张不透明时，用四角众数色当背景抠掉"""
    m = (a > thr).astype(np.uint8)
    if bgr is not None and m.mean() > 0.98:
        h, w = m.shape
        corners = [bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]]
        bg = np.median(np.array(corners, dtype=np.int16), axis=0)
        diff = np.abs(bgr.astype(np.int16) - bg).sum(axis=2)
        m = (diff > 36).astype(np.uint8)
        # 去掉细小噪点
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m


def components(mask, min_area=4):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({"area": int(area), "bbox": [int(x), int(y), int(w), int(h)],
                    "cx": float(cents[i][0]), "cy": float(cents[i][1])})
    out.sort(key=lambda d: -d["area"])
    return out


def edges_touched(mask):
    h, w = mask.shape
    return {
        "top": bool(mask[0, :].any()), "bottom": bool(mask[h - 1, :].any()),
        "left": bool(mask[:, 0].any()), "right": bool(mask[:, w - 1].any()),
    }


def margins(mask):
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return {"left": int(xs.min()), "right": int(w - 1 - xs.max()),
            "top": int(ys.min()), "bottom": int(h - 1 - ys.max())}


def palette(bgr, mask, topk=6):
    px = bgr[mask > 0]
    if len(px) == 0:
        return {"colors": 0, "top": []}
    q = (px // 8 * 8)
    uniq, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)[:topk]
    top = [{"rgb": [int(c[2]), int(c[1]), int(c[0])], "pct": round(float(counts[i]) / len(px) * 100, 1)}
           for i, c in zip(order, uniq[order])]
    return {"colors": int(len(uniq)), "top": top}


def symmetry(mask):
    """左右对称性：镜像 IoU（对 1px 错位敏感）+ 左/右面积比 + 重心偏移（更稳）"""
    flip = mask[:, ::-1]
    inter = int((mask & flip).sum())
    union = int((mask | flip).sum())
    h, w = mask.shape
    left = int(mask[:, :w // 2].sum())
    right = int(mask[:, w - left * 0 - w // 2:].sum())
    ys, xs = np.where(mask > 0)
    cxo = float(xs.mean()) if len(xs) else w / 2.0
    return {
        "iou": round(inter / max(1, union), 3),
        "lr_ratio": round(left / max(1, right), 3),
        "cx_offset_pct": round(abs(cxo - (w - 1) / 2.0) / w * 100, 2),
    }


def grid_check(bgr, mask, cell):
    """按网格看每个格子里颜色是否唯一（像素粒度是否统一）"""
    h, w = mask.shape
    mixed = total = 0
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            blk = bgr[y:y + cell, x:x + cell]
            m = mask[y:y + cell, x:x + cell]
            if m.sum() < cell * cell * 0.6:
                continue
            q = (blk.reshape(-1, 3) // 16)
            if len(np.unique(q, axis=0)) > 2:
                mixed += 1
            total += 1
    return {"cells": total, "mixed": mixed,
            "mixed_pct": round(100.0 * mixed / total, 1) if total else 0.0}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    path = args[0]
    ref = None
    grid = 0
    as_json = "--json" in args
    if "--ref" in args:
        ref = args[args.index("--ref") + 1]
    if "--grid" in args:
        grid = int(args[args.index("--grid") + 1])

    bgr, a = load(path)
    mask = alpha_mask(a, bgr=bgr)
    h, w = mask.shape
    rep = {"file": path, "size": [w, h], "has_alpha": bool((a < 255).any()),
           "opaque_pct": round(100.0 * mask.sum() / (w * h), 1)}
    mg = margins(mask)
    rep["margins"] = mg
    rep["touched"] = edges_touched(mask)
    comps = components(mask)
    rep["components"] = len(comps)
    rep["largest"] = comps[:6]
    rep["palette"] = palette(bgr, mask)
    if grid:
        rep["grid"] = dict(cell=grid, **grid_check(bgr, mask, grid))
    if "--sym" in args:
        rep["symmetry"] = symmetry(mask)
    if ref:
        rbgr, ra = load(ref)
        rmask = alpha_mask(ra, bgr=rbgr)
        rh, rw = rmask.shape
        inter = int((mask > 0) & (cv2.resize(rmask, (w, h), interpolation=cv2.INTER_NEAREST) > 0))
        union = int(((mask > 0) | (cv2.resize(rmask, (w, h), interpolation=cv2.INTER_NEAREST) > 0)))
        rep["ref"] = {"size": [rw, rh], "iou": round(inter / max(1, union), 3)}

    if as_json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return

    print("== %s" % path)
    print("尺寸 %dx%d  带alpha=%s  不透明 %.1f%%" % (w, h, rep["has_alpha"], rep["opaque_pct"]))
    if mg:
        print("留白 左%d 右%d 上%d 下%d" % (mg["left"], mg["right"], mg["top"], mg["bottom"]))
    t = [k for k, v in rep["touched"].items() if v]
    if "--scene" in args:
        print("铺满画面的场景图，跳过贴边判断")
        t = []
    else:
        print("贴边/出框: %s" % ("、".join(t) if t else "无"))
    print("连通域 %d 个（最大 %s）" % (len(comps), comps[0]["bbox"] if comps else "-"))
    for i, c in enumerate(comps[:4]):
        print("   #%d 面积 %d bbox %s" % (i + 1, c["area"], c["bbox"]))
    print("色数 %d，主色 %s" % (rep["palette"]["colors"],
                             ", ".join("rgb%s %.0f%%" % (d["rgb"], d["pct"]) for d in rep["palette"]["top"][:4])))
    if grid:
        g = rep["grid"]
        print("网格 %dpx：%d 格里 %d 格混色（%.1f%%）" % (grid, g["cells"], g["mixed"], g["mixed_pct"]))
    if "--sym" in args:
        sy = rep["symmetry"]
        print("对称性 镜像IoU %.3f 左右面积比 %.2f 重心偏移 %.2f％（正/背视图：面积比 0.9~1.1、偏移<1.5％）"
              % (sy["iou"], sy["lr_ratio"], sy["cx_offset_pct"]))
    if ref:
        print("与参考图 IoU %.3f（尺寸 %dx%d）" % (rep["ref"]["iou"], *rep["ref"]["size"]))


if __name__ == "__main__":
    main()
