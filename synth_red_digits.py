"""
=====================================================================
synth_red_digits.py — 赤い小数桁専用の「合成画像」を作るスクリプト
=====================================================================

■ このファイルの役割
  synth_digits.py は黒桁（黒地に白字）向けの合成データを作る。赤い小数桁は書体も見た目も違う
  （細い・直線的な工業用の数字が、白い箱の中に赤で書かれ、箱の枠線・目盛り帯・▶ が付く）ので、
  赤桁専用のデータを別に作る。学習した CNN は digit_cnn_red.onnx になる。

■ 注意: 字形は「一般的な角型数字」を手で設計したもの
  現物の書体は未確認（実写は meter4 の 2→3 の 1 枚だけで、その字は角が大きく丸い）。
  写真が手に入ったら glyph_paths の座標と角の様式の比率（CORNER_STYLE_WEIGHTS）を直す。

■ 作り方
  1. インクのマスク帯（d-2〜d+2 を縦に並べたもの）を、75% は角型数字、25% は既存フォント字形で描く
  2. 箱の幅・窓の高さで切り出す（静止 / 回転中の位置の決め方は synth_digits.sample と同じ）
  3. 「白い箱に赤字」のカラーに合成し、箱の外（暗い地・▶・目盛り帯・黒桁の切れ端）を付ける
  4. カラーのまま汚す（augment_color。synth_digits.augment はグレー専用なので使えない）
  5. 28×40 に縮小（推論と同じ INTER_AREA）→ red_input.red_cell_input をかけて PNG で保存
     （推論は「カラーのセルを切り出す → 28×40 に縮小 → red_cell_input」なので同じ順にそろえる）

■ 出力
  data/synth_red/<ラベル>/00000.png ... と、書体の記録 kinds.csv（label,index,kind。kind=square|font）
  ラベルの意味は synth_digits.py と同じ（0〜9 静止、10+d は d→d+1 の回転中）

■ 使い方
  python synth_red_digits.py                                            # 学習用（5 万枚）
  python synth_red_digits.py --out data/synth_red_test --n 200 --seed 12345   # 評価用
"""

import os, csv, random, argparse, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

import synth_digits as sd                  # FONTS, PITCH, W, H, drum_strip を再利用（import してもフォント一覧が出るだけ）
from red_input import red_cell_input

# ================== 設定値 ==================
OUT = "data/synth_red"       # 出力先フォルダ
N_PER_CLASS = 2500           # 1 ラベルあたりの枚数（20 × 2500 = 5 万枚。synth_digits.py と同数）
FONT_GLYPH_RATE = 0.25       # 既存フォント字形を混ぜる割合。角型だけだと角型に過学習するし、
                             # 現物が角型でなかった場合の保険にもなる
PITCH = sd.PITCH             # ドラム上での数字の間隔（大きめに描いて最後に縮小する）
CORNER_STYLE_WEIGHTS = [0.2, 0.25, 0.55]   # 角の様式 sharp / chamfer / round の比率。現物未確認なので全部出す。
                                           # meter4 の 2・3 は角丸なので round を多めにした（推測。要検証）


# ================== 角型数字の描画 ==================
# 各数字 = ストローク（折れ線）のリスト。点は (x, y, 丸めてよいか)。x: 0=左 1=右, y: 0=上 1=下
# 「丸めてよいか」False の角は常に尖らせる（5 の左上、7 の右上など、丸めると別の数字に寄る角）
def glyph_paths(d, rng):
    ym = rng.uniform(0.44, 0.52)                     # 中段の高さ（3, 5, 6, 8, 9 の腰）
    if d == 0:
        return [[(0.5, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0.5, 0, 0)]]   # 閉じた四角。辺の中点から始めて 4 つの角をすべて丸める
    if d == 1:
        xs = rng.uniform(0.55, 0.7)
        p = [[(xs, 0, 0), (xs, 1, 0)]]
        fy = rng.uniform(0.12, 0.25)
        if rng.random() < 0.6:
            p.append([(xs, 0, 0), (xs - rng.uniform(0.3, 0.45), fy, 0)])        # 斜めの旗
        else:
            p.append([(xs, 0, 0), (xs - rng.uniform(0.25, 0.4), 0, 0)])         # 水平の旗
        if rng.random() < 0.25:
            p.append([(0.15, 1, 0), (0.95, 1, 0)])                              # 足（台座）
        return p
    if d == 2:
        y0 = rng.uniform(0.15, 0.3)
        if rng.random() < 0.8:                       # 上の箱から斜めに降りる
            yk = rng.uniform(0.3, 0.45)
            return [[(0, y0, 0), (0, 0, 1), (1, 0, 1), (1, yk, 1), (0, 1, 0), (1, 1, 0)]]
        return [[(0, y0, 0), (0, 0, 1), (1, 0, 1), (1, ym, 1), (0, ym, 1), (0, 1, 0), (1, 1, 0)]]   # 直角に折れる
    if d == 3:
        if rng.random() < 0.5:                       # 上が平ら（斜めで中段へ）
            xm = rng.uniform(0.3, 0.5)
            return [[(0, 0, 0), (1, 0, 0), (xm, ym, 0), (1, ym, 1), (1, 1, 1), (0, 1, 1), (0, rng.uniform(0.75, 0.88), 0)]]
        y0 = rng.uniform(0.12, 0.25)                 # 上も箱型（丸めると meter4 の丸い 3 に近い）
        return [[(0, y0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 1 - y0, 0)],
                [(rng.uniform(0.25, 0.45), ym, 0), (1, ym, 0)]]
    if d == 4:
        yb = rng.uniform(0.62, 0.72)                 # 横棒の高さ
        xs = rng.uniform(0.62, 0.78)                 # 縦棒の x
        u = rng.random()
        if u < 0.4:                                  # 閉じた 4（斜線が縦棒の上端に届く）
            return [[(xs, 1, 0), (xs, 0, 0), (0, yb, 0), (1, yb, 0)]]
        if u < 0.75:                                 # 上が開いた 4（左の縦線 + 横棒、縦棒は横棒より上から）
            return [[(0, 0, 0), (0, yb, 0), (1, yb, 0)], [(xs, rng.uniform(0.25, 0.45), 0), (xs, 1, 0)]]
        g = rng.uniform(0.05, 0.15)                  # 上が開いた 4（斜線と縦棒の上端にすき間）
        return [[(xs - g, g * 0.5, 0), (0, yb, 0), (1, yb, 0)], [(xs, 0, 0), (xs, 1, 0)]]
    if d == 5:
        return [[(1, 0, 0), (0, 0, 0), (0, ym, 0), (1, ym, 1), (1, 1, 1), (0, 1, 1), (0, rng.uniform(0.8, 0.9), 0)]]
    if d in (6, 9):
        if rng.random() < 0.7:                       # 直線の 6（左の縦棒 + 下の箱）
            p = [[(rng.uniform(0.8, 1.0), 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, ym, 1), (0, ym, 0)]]
        else:                                        # 斜めの 6（右上から斜めに降りて下の箱）
            p = [[(rng.uniform(0.7, 0.9), 0, 0), (0, ym, 0), (0, 1, 1), (1, 1, 1), (1, ym, 1), (0, ym, 0)]]
        if d == 9:                                   # 9 は 6 を 180° 回したもの
            p = [[(1 - x, 1 - y, f) for x, y, f in s] for s in p]
        return p
    if d == 7:
        p = [[(1, 0, 0), (rng.uniform(0.25, 0.45), 1, 0)]]
        if rng.random() < 0.5:                       # 右上で少し垂直に降りてから斜め
            p = [[(1, 0, 0), (1, rng.uniform(0.1, 0.25), 1), (rng.uniform(0.3, 0.45), 1, 0)]]
        top = [(0, 0, 0), (1, 0, 0)]
        if rng.random() < 0.35:
            top = [(0, rng.uniform(0.15, 0.25), 0)] + top   # 左端が下がる
        return [top] + p
    if d == 8:
        k = rng.uniform(0.04, 0.1)                   # 上の箱を少し細くする
        return [[(0.5, 0, 0), (1 - k, 0, 1), (1 - k, ym, 1), (k, ym, 1), (k, 0, 1), (0.5, 0, 0)],      # 上の箱（辺の中点から一周）
                [(0.5, ym, 0), (1, ym, 1), (1, 1, 1), (0, 1, 1), (0, ym, 1), (0.5, ym, 0)]]          # 下の箱


def _round_path(pts, r, style, steps=6):
    """折れ線の内側の角を半径 r（画素）で丸める / 面取りする。pts: [(x, y, 丸めてよいか)]"""
    out = [pts[0][:2]]
    for i in range(1, len(pts) - 1):
        (x0, y0, _), (x1, y1, ok), (x2, y2, _) = pts[i - 1], pts[i], pts[i + 1]
        if style == "sharp" or not ok or r <= 0:
            out.append((x1, y1)); continue
        l1, l2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        rr = min(r, l1 * 0.5, l2 * 0.5)              # 辺の半分を超えない（短い辺で折り返さない）
        a = (x1 - (x1 - x0) / l1 * rr, y1 - (y1 - y0) / l1 * rr)   # 角の手前
        b = (x1 + (x2 - x1) / l2 * rr, y1 + (y2 - y1) / l2 * rr)   # 角の先
        if style == "chamfer":
            out += [a, b]
        else:                                        # 2 次ベジエで丸める（円弧の近似で十分）
            for t in [j / steps for j in range(steps + 1)]:
                out.append(((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * x1 + t * t * b[0],
                            (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * y1 + t * t * b[1]))
    out.append(pts[-1][:2])
    return out


def draw_square_digit(dr, d, cx, cy, h, fg, rng=random, width_ratio=None, stroke=None, corner=None, style=None):
    """
    角型数字 d を中心 (cx, cy)・高さ h で描く。
      字幅   = h * 0.50〜0.65（赤桁は細身。meter4 の実測 ≈ 0.62）
      線幅   = h * 0.09〜0.15（meter4 の実測 ≈ 0.12。黒桁の合成 0.13〜0.17 より細め）
      角     = sharp 20% / chamfer 25% / round 55%、半径 = 字幅 * 0〜0.45
    線の中心が外接枠に乗るので、外形は線幅の半分だけはみ出す → 枠を線幅の半分だけ内側に取る
    """
    wr = width_ratio if width_ratio is not None else rng.uniform(0.50, 0.65)
    t = stroke if stroke is not None else max(2, int(round(h * rng.uniform(0.09, 0.15))))
    st = style or rng.choices(["sharp", "chamfer", "round"], CORNER_STYLE_WEIGHTS)[0]
    w = h * wr
    x0, y0 = cx - w / 2 + t / 2, cy - h / 2 + t / 2
    sw, sh = w - t, h - t
    r = (corner if corner is not None else rng.uniform(0.0, 0.45)) * w
    for path in glyph_paths(d, rng):
        pts = [(x0 + x * sw, y0 + y * sh, f) for x, y, f in path]
        dr.line(_round_path(pts, r, st), fill=fg, width=t, joint="curve")


# ================== 帯・窓・カラー合成 ==================
def red_strip(d, cw):
    """
    d-2..d+2 を縦に並べたインクのマスク帯（L, インク=255）を作る。
    返り値: (帯, 中央の数字の中心 y, 字の高さ, 書体 "square" か "font")
    """
    # 角型の字高 / ピッチ。meter4 実測 ≈ 0.82（黒桁の 0.62〜0.95 より詰まっている）。
    # フォント字形にも同じ size を渡すが、これはフォントサイズなので実際の字高は約 0.7 倍になり、角型より一回り小さい。
    # 学習済みモデルがこの分布で作られているので生成は変えない（現物が角型でない場合の保険として、小さめも許容）
    size = int(PITCH * random.uniform(0.72, 0.92))
    if random.random() < FONT_GLYPH_RATE:
        font = random.choice(sd.FONTS)
        # 赤桁は細いので太らせない（stroke 0〜2）。平らな 3・直線の 6/9・開いた 4 も drum_strip の中で混ざる
        strip, yc = sd.drum_strip(d, font, size, 255, 0, random.choice([0, 0, 0, 1, 2]), cw)
        return strip, yc, size, "font"
    strip = Image.new("L", (cw, PITCH * 5), 0)
    dr = ImageDraw.Draw(strip)
    # 1 本の帯の中では書体のパラメータをそろえる（同じドラムの数字は同じ書体。
    # 回転中の上下 2 字が別書体になると、実物にない組み合わせを覚えてしまう）
    wr = random.uniform(0.50, 0.65)                                       # 字幅 / 字高（meter4 実測 0.62）
    t = max(2, int(round(size * random.uniform(0.09, 0.15))))             # 線幅 / 字高（meter4 実測 0.12）
    st = random.choices(["sharp", "chamfer", "round"], CORNER_STYLE_WEIGHTS)[0]
    corner = random.uniform(0.0, 0.45)                                    # 角の半径 / 字幅。0.45 で 2・3 の上が半円に近づく
    dx = random.uniform(-cw * 0.06, cw * 0.06)                            # 字の横位置（synth_digits.drum_strip と同じ）
    tick = random.random() < 0.4                                          # 箱の右端の赤い目盛り（meter4 では各数字の中段の高さ。ドラムと一緒に動く）
    for i, k in enumerate(range(d - 2, d + 3)):
        cy = (i + 0.5) * PITCH
        draw_square_digit(dr, k % 10, cw / 2 + dx, cy, size, 255, width_ratio=wr, stroke=t, corner=corner, style=st)
        if tick:
            x1 = cw - 1
            dr.rectangle((x1 - t * 1.2, cy - t / 2, x1, cy + t / 2), fill=255)
    return strip, 2.5 * PITCH, size, "square"


def sample_mask(label):
    """窓のインクマスク（L）と書体を返す。静止 / 回転中の位置の決め方は synth_digits.sample と同じ"""
    d = label % 10
    cw = int(PITCH * random.uniform(0.62, 0.8))           # 箱の内幅。meter4 は箱 37px / ピッチ 51px ≈ 0.73
    strip, yc, _, kind = red_strip(d, cw)
    win_h = int(PITCH * random.uniform(0.95, 1.6))        # 箱の高さ。meter4 は ≈ 1.5 ピッチ
    frac = random.uniform(-0.12, 0.12) if label < 10 else random.uniform(0.33, 0.67)   # meter4 の回転位置は 0.53
    top = int(yc + frac * PITCH - win_h / 2)
    return strip.crop((0, top, cw, top + win_h)), kind


def compose(mask):
    """インクマスクを「白い箱に赤字」のカラー（RGB）に合成し、箱の外（暗い地・▶・目盛り帯）を付ける"""
    bw, bh = mask.size
    m = np.asarray(mask, np.float32) / 255.0
    paper = np.array([random.uniform(200, 250)] * 3) + np.random.uniform(-10, 10, 3)
    # インク: meter4 は R126 G48 B53 だが、露出で明るくなるので R は上側も取る
    ink = np.array([random.uniform(150, 235), random.uniform(20, 90), 0.0])
    ink[2] = np.clip(ink[1] + random.uniform(-20, 20), 10, 110)
    box = paper[None, None] * (1 - m[..., None]) + ink[None, None] * m[..., None]
    # 箱の右端の目盛り帯（白黒の縦の市松）: 60%
    if random.random() < 0.6:
        sw = int(bw * random.uniform(0.08, 0.16)); per = int(bh / random.uniform(4, 7)) or 1; ph = random.randint(0, per)
        dark = random.uniform(15, 60)
        for y in range(bh):
            if ((y + ph) // per) % 2 == 0:
                box[y, bw - sw:] = dark
    # セル全体の幅: 推論のセル幅は黒桁のピッチなので箱より広い（箱はセルの 60〜90%。meter4 は約 75%）
    Wc = int(bw / random.uniform(0.6, 0.9)); Hc = bh
    out_col = random.uniform(10, 70)                      # 箱の外は暗い灰（黒い文字盤）
    cell = np.full((Hc, Wc, 3), out_col, np.float32)
    x0 = int((Wc - bw) / 2 + random.uniform(-0.15, 0.15) * Wc)
    x0 = int(np.clip(x0, -bw * 0.15, Wc - bw * 0.85))
    xa, xb = max(0, x0), min(Wc, x0 + bw)
    cell[:, xa:xb] = box[:, xa - x0:xb - x0]
    img = Image.fromarray(np.clip(cell, 0, 255).astype(np.uint8), "RGB")
    dr = ImageDraw.Draw(img)
    # 箱の枠線（暗い線）: 70%。消えるケースも残す
    if random.random() < 0.7:
        dr.rectangle((x0, -5, x0 + bw - 1, Hc + 5), outline=(int(out_col),) * 3, width=random.randint(1, 5))
    # 上下に箱の外の黒がのぞく: 50%
    if random.random() < 0.5:
        for yy in ((0, int(Hc * random.uniform(0, 0.12))), (Hc - int(Hc * random.uniform(0, 0.12)), Hc)):
            dr.rectangle((0, yy[0], Wc, yy[1]), fill=(int(out_col),) * 3)
    # 左に黒桁の白い字の切れ端: 30%
    if random.random() < 0.3 and x0 > 4:
        g = int(random.uniform(150, 240))
        xr = random.uniform(0, x0 * 0.8)
        dr.rectangle((xr - Wc * 0.08, Hc * random.uniform(0.15, 0.4), xr, Hc * random.uniform(0.6, 0.85)), fill=(g, g, g))
    # 右に「▶」の一部（白い三角）: 30%
    if random.random() < 0.3:
        g = int(random.uniform(180, 245)); xl = x0 + bw + random.uniform(-2, 6)
        cy = Hc * random.uniform(0.35, 0.65); s = Hc * random.uniform(0.12, 0.25)
        dr.polygon([(xl, cy), (xl + s * 1.6, cy - s), (xl + s * 1.6, cy + s)], fill=(g, g, g))
    return img, int(out_col)


def augment_color(img, out_col):
    """
    synth_digits.augment のカラー版。幾何と明るさの範囲は同じで、色かぶり・彩度低下・JPEG を足す。
    最後は推論の _cut_cell と同じ cv2.INTER_AREA で 28×40 の BGR にする
    （synth_digits.augment は最後に LANCZOS で縮小するグレー専用なので、そのまま使えない）
    """
    fill = (out_col,) * 3
    img = img.rotate(random.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=fill)
    if random.random() < 0.5:                                                     # せん断
        img = img.transform(img.size, Image.AFFINE, (1, random.uniform(-0.15, 0.15), 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=fill)
    if random.random() < 0.5:                                                     # 横ずらし
        img = img.transform(img.size, Image.AFFINE, (1, 0, int(img.size[0] * random.uniform(-0.12, 0.12)), 0, 1, 0), resample=Image.BILINEAR, fillcolor=fill)
    if random.random() < 0.7:                                                     # ぼかし
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 3.5)))
    a = np.asarray(img).astype(np.float32)
    if random.random() < 0.5:                                                     # 彩度低下（くすんだ赤・逆光）
        gray = a.mean(2, keepdims=True); a = a + (gray - a) * random.uniform(0, 0.7)
    a *= np.random.uniform(0.85, 1.15, 3)[None, None]                             # ホワイトバランスのずれ（暖色・寒色かぶり）
    a = (a - 128) * random.uniform(0.35, 1.2) + 128 + random.uniform(-40, 40)     # コントラストと明るさ
    if random.random() < 0.3:                                                     # ガンマ
        a = 255 * (np.clip(a, 0, 255) / 255) ** random.uniform(0.5, 2.0)
    if random.random() < 0.5:                                                     # 明るさのグラデーション
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        g = xx * random.uniform(-1, 1) + yy * random.uniform(-1, 1); g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        a += (g * random.uniform(20, 90))[..., None]
    if random.random() < 0.3:                                                     # 丸いハイライト（反射）
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        cx, cy = random.uniform(0, a.shape[1]), random.uniform(0, a.shape[0]); r = random.uniform(0.2, 0.6) * max(a.shape[:2])
        a += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * random.uniform(30, 100))[..., None]
    a += np.random.normal(0, random.uniform(2, 15), a.shape)                      # センサーノイズ（チャネルごと独立）
    a = np.clip(a, 0, 255).astype(np.uint8)
    bgr = a[..., ::-1].copy()                                                     # PIL の RGB → OpenCV の BGR（red_cell_input は BGR 前提）
    if random.random() < 0.3:
        # JPEG: 色差の間引きで赤の縁がにじむのを再現する。セルの 2〜3 倍の大きさで圧縮するだけにする
        # （セルより粗い縮小は、低解像度化 augmentation として README で悪化して戻した）
        s = random.uniform(2, 3)
        small = cv2.resize(bgr, (int(sd.W * s), int(sd.H * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(bgr, (sd.W, sd.H), interpolation=cv2.INTER_AREA)


def make(label):
    """1 枚作る。返り値: (28×40 の BGR, 書体)。保存するときは red_cell_input をかける"""
    mask, kind = sample_mask(label)
    img, oc = compose(mask)
    return augment_color(img, oc), kind


# ================== メイン処理 ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="出力先フォルダ")
    ap.add_argument("--n", type=int, default=N_PER_CLASS, help="1 ラベルあたりの枚数")
    ap.add_argument("--seed", type=int, default=0, help="乱数の種（評価用は 12345 など学習と別の値にする）")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    rows = []
    for label in range(20):
        os.makedirs(f"{a.out}/{label}", exist_ok=True)
        for i in range(a.n):
            bgr, kind = make(label)
            Image.fromarray(red_cell_input(bgr)).save(f"{a.out}/{label}/{i:05d}.png")
            rows.append((label, i, kind))
        print("class", label, "done", flush=True)
    with open(f"{a.out}/kinds.csv", "w", newline="") as f:      # 書体別の正解率を出すための記録
        w = csv.writer(f); w.writerow(["label", "index", "kind"]); w.writerows(rows)


if __name__ == "__main__":
    main()
