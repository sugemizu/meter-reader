"""
ドラム式カウンタの数字を合成生成する。
縦に 0..9 が並んだ「ドラム帯」を描き、窓を切り出すことで
・上下に隣の数字がのぞく（静止）
・2つの数字が上下に割れる（回転中）
を物理的に同じ仕組みで生成する。

出力: data/synth/{label}/*.png
  label 0-9   : 静止
  label 10+d  : d → d+1 の回転中（読み値は小さい方 d）
"""
import os, random, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = "data/synth"
W, H = 28, 40              # 1桁の入力サイズ（縦長）
N_PER_CLASS = 2500
REVERSE_ROLL_RATE = 0.0    # 逆方向に回る機種がある場合のみ >0 に
PITCH = 120                # ドラム上の数字間隔（大きめに描いて後で縮小）

def find_fonts():
    cands = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) + glob.glob("fonts/*.ttf")
    keys = ("sans-bold", "sansbold", "poppins-bold", "poppins-medium", "carlito-bold", "arial", "roboto", "helvetica")
    pref = [f for f in cands if any(k in f.lower() for k in keys) and "italic" not in f.lower() and "oblique" not in f.lower()]
    return pref or cands

FONTS = find_fonts()
assert FONTS, "ttfフォントが見つかりません。fonts/ に入れてください"

def draw_flat_top_3(dr, cx, cy, h, fg, width):
    """
    メーターによくある「上が平らな3」: 水平バー → 斜線で中央へ → 下半分は丸い弓。
    フォントの丸い3と混ぜて学習させる。
    """
    w = h * 0.55
    x0, x1 = cx - w / 2, cx + w / 2
    yt, ym, yb = cy - h / 2, cy, cy + h / 2
    dr.line([(x0, yt), (x1, yt)], fill=fg, width=width)
    dr.line([(x1, yt), (cx - w * 0.05, ym)], fill=fg, width=width)
    # 下の弓: 中央から右へ出て、下で左へ戻る
    r = (yb - ym) / 2
    box = (x1 - 2 * r, ym, x1, yb)
    dr.arc(box, start=-90, end=170, fill=fg, width=width)
    dr.line([(cx - w * 0.05, ym), (x1 - r, ym)], fill=fg, width=width)

FLAT3_RATE = 0.5

def drum_strip(d, font, size, fg, bg, stroke, cw):
    """
    d-2, d-1, d, d+1, d+2 を縦に並べたドラム帯。
    上が d-1（通過済み）、下が d+1（次）。数字 k の中心 y = (k_idx + 0.5) * PITCH
    返り値: (帯画像, d の中心 y)
    """
    strip = Image.new("L", (cw, PITCH * 5), bg)
    dr = ImageDraw.Draw(strip)
    f = ImageFont.truetype(font, size)
    use_flat3 = random.random() < FLAT3_RATE
    # 数字の実高さ（フォントにより size と異なる）をそろえるため 0 の高さを基準にする
    b0 = dr.textbbox((0, 0), "0", font=f, stroke_width=stroke)
    glyph_h = b0[3] - b0[1]
    for i, k in enumerate(range(d - 2, d + 3)):
        s = str(k % 10)
        cx = cw / 2 + random.uniform(-cw * 0.06, cw * 0.06)
        cy = (i + 0.5) * PITCH
        if s == "3" and use_flat3:
            draw_flat_top_3(dr, cx, cy, glyph_h, fg, max(4, int(glyph_h * 0.13) + stroke))
            continue
        bbox = dr.textbbox((0, 0), s, font=f, stroke_width=stroke)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        dr.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), s, fill=fg, font=f, stroke_width=stroke, stroke_fill=fg)
    # ドラム間の暗い継ぎ目（あったりなかったり）
    if random.random() < 0.5:
        g = random.randint(1, 4)
        for i in range(1, 5):
            y = i * PITCH
            dr.rectangle((0, y - g // 2, cw, y + g // 2), fill=min(fg, bg))
    return strip, 2.5 * PITCH

def sample(label, font, fg, bg):
    d = label if label < 10 else label - 10
    size = int(PITCH * random.uniform(0.62, 0.95))   # 実機は数字がピッチの8割以上を占める
    stroke = random.choice([0, 0, 1, 2, 3, 4, 5])
    cw = int(PITCH * random.uniform(0.6, 0.85))
    strip, yc = drum_strip(d, font, size, fg, bg, stroke, cw)
    # 窓の高さ: 1ピッチ〜1.5ピッチ（隣がのぞく量に相当）
    win_h = int(PITCH * random.uniform(0.95, 1.5))
    if label < 10:
        frac = random.uniform(-0.12, 0.12)           # 静止（わずかなズレ）
    else:
        frac = random.uniform(0.33, 0.67)            # 回転中: d と d+1 がほぼ半々
        if random.random() < REVERSE_ROLL_RATE:
            frac = -frac
    # 回転方向: d が上へ抜け d+1 が下から来る → 窓の中心は d+1 側（下）へ移動
    cy = yc + frac * PITCH
    top = int(cy - win_h / 2)
    win = strip.crop((0, top, cw, top + win_h))
    return win

def augment(img):
    bg = int(np.array(img)[0, 0])
    img = img.rotate(random.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=bg)
    if random.random() < 0.5:
        sh = random.uniform(-0.15, 0.15)
        img = img.transform(img.size, Image.AFFINE, (1, sh, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=bg)
    # 左右の切れ（隣のセルの縁が入る / 自分が少し切れる）
    if random.random() < 0.5:
        w, h = img.size
        dx = int(w * random.uniform(-0.12, 0.12))
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, 0), resample=Image.BILINEAR, fillcolor=bg)
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 3.5)))
    arr = np.asarray(img).astype(np.float32)
    arr = (arr - 128) * random.uniform(0.35, 1.2) + 128 + random.uniform(-40, 40)
    if random.random() < 0.5:   # 反射・照明ムラ
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        g = xx * random.uniform(-1, 1) + yy * random.uniform(-1, 1)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        arr += g * random.uniform(20, 90)
    if random.random() < 0.3:   # 局所的なハイライト
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        cx, cy = random.uniform(0, arr.shape[1]), random.uniform(0, arr.shape[0])
        r = random.uniform(0.2, 0.6) * max(arr.shape)
        arr += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * random.uniform(30, 100)
    arr += np.random.normal(0, random.uniform(2, 15), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if random.random() < 0.25:  # 小数点・隣セルのドットが角に入る
        dr = ImageDraw.Draw(img)
        w, h = img.size
        r = random.randint(2, max(3, w // 8))
        cx = random.choice([random.randint(0, r), random.randint(w - r, w)])
        cy = random.randint(int(h * 0.55), h)
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=random.randint(150, 255))
    if random.random() < 0.3:   # 窓枠の縁
        dr = ImageDraw.Draw(img)
        w, h = img.size
        side = random.choice(["l", "r", "t", "b"])
        t = random.randint(2, max(3, w // 10))
        box = {"l": (0, 0, t, h), "r": (w - t, 0, w, h), "t": (0, 0, w, t), "b": (0, h - t, w, h)}[side]
        dr.rectangle(box, fill=random.randint(0, 255))
    return img.resize((W, H), Image.LANCZOS)

def main():
    for label in range(20):
        os.makedirs(f"{OUT}/{label}", exist_ok=True)
    for label in range(20):
        for i in range(N_PER_CLASS):
            font = random.choice(FONTS)
            if random.random() < 0.8:
                fg, bg = random.randint(170, 255), random.randint(0, 70)     # 黒地白字
            else:
                fg, bg = random.randint(0, 70), random.randint(170, 255)     # 白地黒字 / 赤地
            augment(sample(label, font, fg, bg)).save(f"{OUT}/{label}/{i:05d}.png")
        print("class", label, "done")

if __name__ == "__main__":
    main()
