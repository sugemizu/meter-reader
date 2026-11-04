"""
=====================================================================
synth_digits.py — 学習用の「合成画像」を作るスクリプト
=====================================================================

■ このファイルの役割
  実写の水道メーター画像がほとんど無いので、代わりにプログラムで
  「それっぽい数字の画像」を大量に作ります。これが CNN の教材になります。

■ 発想
  メーターの数字は、縦に 0〜9 が並んだ「ドラム（円筒）」が回って表示されます。
  そこで本物と同じように
    1. 縦に数字が並んだ細長い帯（ドラム帯）を描き
    2. その一部を「窓」で切り出す
  という作り方をします。こうすると
    ・静止中  … 窓の中央に数字、上下に隣の数字がちょっと見える
    ・回転中  … 2 つの数字が上下に割れて見える
  がどちらも自然に作れます。

■ 出力
  data/synth/<ラベル>/00000.png ... という形で保存します。
    ラベル 0〜9  : 静止している数字
    ラベル 10〜19: 「d → d+1 に回転中」（10+d）。読み値は小さい方 d を採用

■ 使い方
  python synth_digits.py
"""

# ---- 標準ライブラリ ----
import os        # フォルダ作成など OS 操作
import random    # 乱数（ばらつきを作る）
import glob      # ファイル名のパターン検索（フォント探しに使う）

# ---- 外部ライブラリ ----
import numpy as np   # 画像を数値の配列として扱う
from PIL import Image, ImageDraw, ImageFont, ImageFilter   # 画像の描画・加工（Pillow）

# ================== 設定値 ==================
OUT = "data/synth"         # 出力先フォルダ
W, H = 28, 40              # 1桁の画像サイズ（幅28 × 高さ40 ピクセル）。CNN の入力サイズと一致させる
N_PER_CLASS = 2500         # 1 ラベルあたり何枚作るか（20ラベル × 2500 = 5万枚）
REVERSE_ROLL_RATE = 0.0    # ドラムが逆方向に回る機種がある場合だけ 0 より大きくする（通常は 0）
PITCH = 120                # ドラム上での数字と数字の間隔（ピクセル）。大きめに描いて最後に縮小する
FLAT3_RATE = 0.5           # 「上が平らな3」を使う確率（メーターの3はこの形が多い）


# ================== フォントを探す ==================
def find_fonts():
    """
    PC に入っている太めのサンセリフ体フォント（.ttf）を探して、パスの一覧を返す。
    Windows/Mac などで見つからない場合は、プロジェクト内の fonts/ フォルダにも探しに行く。
    """
    # /usr/share/fonts 以下を再帰的に検索 + fonts/ フォルダも検索
    cands = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) + glob.glob("fonts/*.ttf")
    # ファイル名にこれらの文字列を含むものを「太めのサンセリフ」とみなす
    keys = ("sans-bold", "sansbold", "poppins-bold", "poppins-medium", "carlito-bold", "arial", "roboto", "helvetica")
    # 条件に合い、かつ斜体（italic / oblique）でないものだけ残す
    pref = [f for f in cands
            if any(k in f.lower() for k in keys)
            and "italic" not in f.lower()
            and "oblique" not in f.lower()]
    # 好みのフォントが 1 つも無ければ、見つかった全フォントを使う
    return pref or cands

FONTS = find_fonts()
# フォントが 1 つも無ければエラーで止める（assert = 条件が偽ならエラー）
assert FONTS, "ttfフォントが見つかりません。fonts/ に入れてください"


# ================== 「上が平らな3」を手描きする ==================
def draw_flat_top_3(dr, cx, cy, h, fg, width):
    """
    メーターによくある「上が平らな 3」を線と弧で描く。
    （PC のフォントの 3 は上が丸いので、それだけだと 5 と間違えやすい）

    dr    : ImageDraw オブジェクト（描画用のペン）
    cx, cy: 数字の中心座標
    h     : 数字の高さ
    fg    : 線の色（0〜255 のグレー値）
    width : 線の太さ
    """
    w = h * 0.55                        # 数字の幅は高さの 55%
    x0, x1 = cx - w / 2, cx + w / 2     # 左端・右端の x 座標
    yt, ym, yb = cy - h / 2, cy, cy + h / 2   # 上端・中央・下端の y 座標

    dr.line([(x0, yt), (x1, yt)], fill=fg, width=width)              # 上の水平バー
    dr.line([(x1, yt), (cx - w * 0.05, ym)], fill=fg, width=width)   # 右上から中央へ斜線

    # 下半分の丸い弓（arc = 円弧）。中央から右へ出て、下で左へ戻る
    r = (yb - ym) / 2                   # 弓の半径
    box = (x1 - 2 * r, ym, x1, yb)      # 弓を収める四角形
    dr.arc(box, start=-90, end=170, fill=fg, width=width)
    dr.line([(cx - w * 0.05, ym), (x1 - r, ym)], fill=fg, width=width)   # 中央の短い横線


# ================== ドラム帯を描く ==================
def drum_strip(d, font, size, fg, bg, stroke, cw):
    """
    d-2, d-1, d, d+1, d+2 の 5 つの数字を縦に並べた「ドラム帯」の画像を作る。
    上にある数字が d-1（もう通り過ぎた数字）、下にある数字が d+1（次に来る数字）。

    d      : 中央に置く数字（0〜9）
    font   : フォントファイルのパス
    size   : フォントサイズ
    fg, bg : 文字の色 / 背景の色（グレー値）
    stroke : 文字を太らせる量（0 なら太らせない）
    cw     : 帯の幅

    戻り値: (帯の画像, 中央の数字 d の y 座標)
    """
    # 高さ PITCH×5 の真っさらな帯を作る（"L" = グレースケール画像）
    strip = Image.new("L", (cw, PITCH * 5), bg)
    dr = ImageDraw.Draw(strip)                   # この帯に描くためのペン
    f = ImageFont.truetype(font, size)           # フォントを読み込む
    use_flat3 = random.random() < FLAT3_RATE     # この帯では平らな 3 を使うか？

    # フォントごとに文字の実際の高さが違うので、「0」の高さを基準にそろえる
    b0 = dr.textbbox((0, 0), "0", font=f, stroke_width=stroke)   # 0 を描いたときの枠
    glyph_h = b0[3] - b0[1]                                      # 枠の高さ = 数字の高さ

    # enumerate = 番号付きでループ。i は 0〜4、k は d-2〜d+2
    for i, k in enumerate(range(d - 2, d + 3)):
        s = str(k % 10)                               # k を 0〜9 に丸める（例: -1 → 9, 10 → 0）
        cx = cw / 2 + random.uniform(-cw * 0.06, cw * 0.06)   # 横位置を少しランダムにずらす
        cy = (i + 0.5) * PITCH                        # i 番目の数字の中心 y

        if s == "3" and use_flat3:
            # 3 だけ特別扱い: 手描きの平らな 3
            draw_flat_top_3(dr, cx, cy, glyph_h, fg, max(4, int(glyph_h * 0.13) + stroke))
            continue   # 次の数字へ

        # 通常の数字: フォントで描く
        bbox = dr.textbbox((0, 0), s, font=f, stroke_width=stroke)   # 文字の枠（左, 上, 右, 下）
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]                # 文字の幅・高さ
        # (cx, cy) が文字の中心になるように左上座標を計算して描く
        dr.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), s,
                fill=fg, font=f, stroke_width=stroke, stroke_fill=fg)

    # ドラムとドラムの間に見える暗い継ぎ目を 50% の確率で描く
    if random.random() < 0.5:
        g = random.randint(1, 4)                 # 継ぎ目の太さ
        for i in range(1, 5):
            y = i * PITCH                        # 数字と数字の境目の y
            dr.rectangle((0, y - g // 2, cw, y + g // 2), fill=min(fg, bg))   # 暗い方の色で線

    return strip, 2.5 * PITCH   # 中央の数字（i=2）の中心 y は 2.5×PITCH


# ================== 帯から窓を切り出す ==================
def sample(label, font, fg, bg):
    """
    ラベルに応じた 1 枚の窓画像を作る。

    label : 0〜9 なら静止、10〜19 なら回転中（10+d）
    """
    d = label if label < 10 else label - 10            # 中央に置く数字
    size = int(PITCH * random.uniform(0.62, 0.95))     # 数字の高さ（実機は密なので大きめ）
    stroke = random.choice([0, 0, 1, 2, 3, 4, 5])      # 太らせ量（0 が出やすい）
    cw = int(PITCH * random.uniform(0.6, 0.85))        # 窓の幅

    strip, yc = drum_strip(d, font, size, fg, bg, stroke, cw)   # ドラム帯を描く

    # 窓の高さ: 1ピッチ〜1.5ピッチ。高いほど上下の隣の数字が多くのぞく
    win_h = int(PITCH * random.uniform(0.95, 1.5))

    if label < 10:
        frac = random.uniform(-0.12, 0.12)    # 静止: 中心からわずかにずれる程度
    else:
        frac = random.uniform(0.33, 0.67)     # 回転中: d と d+1 がほぼ半々に見える位置
        if random.random() < REVERSE_ROLL_RATE:
            frac = -frac                      # 逆回転の機種用

    # 回転方向: 現在の数字 d が上へ抜けて、次の d+1 が下から来る。
    # → 窓の中心を d+1 側（下）へ frac×PITCH だけ動かす
    cy = yc + frac * PITCH
    top = int(cy - win_h / 2)                               # 窓の上端
    win = strip.crop((0, top, cw, top + win_h))             # 帯から窓部分を切り出す
    return win


# ================== 実写っぽく汚す（データ拡張） ==================
def augment(img):
    """
    きれいな合成画像に、実写で起きる「ゆがみ・ぼけ・反射・ノイズ」をランダムに加える。
    これをやらないと CNN が「きれいな画像」しか読めなくなる。
    最後に CNN の入力サイズ (W×H) に縮小して返す。
    """
    bg = int(np.array(img)[0, 0])   # 左上ピクセルの色を「背景色」として使う（回転時の余白埋め用）

    # 回転（-8〜+8 度）
    img = img.rotate(random.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=bg)

    # せん断（平行四辺形にゆがめる）: 50% の確率
    if random.random() < 0.5:
        sh = random.uniform(-0.15, 0.15)
        img = img.transform(img.size, Image.AFFINE, (1, sh, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=bg)

    # 左右にずらす（桁の切り出し位置がずれるのを再現）: 50% の確率
    if random.random() < 0.5:
        w, h = img.size
        dx = int(w * random.uniform(-0.12, 0.12))
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, 0), resample=Image.BILINEAR, fillcolor=bg)

    # ぼかし（ピント外れ・手ぶれ）: 70% の確率
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 3.5)))

    # ここから先は数値配列（numpy）として処理する。float32 = 小数で計算
    arr = np.asarray(img).astype(np.float32)

    # コントラストと明るさをランダムに変える
    #   (arr-128)*係数 でコントラスト、+ランダム値 で明るさ
    arr = (arr - 128) * random.uniform(0.35, 1.2) + 128 + random.uniform(-40, 40)

    # 反射・照明ムラ（画面の片側が明るいグラデーション）: 50% の確率
    if random.random() < 0.5:
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]           # 各ピクセルの座標
        g = xx * random.uniform(-1, 1) + yy * random.uniform(-1, 1)  # 斜めの傾き
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)              # 0〜1 に正規化
        arr += g * random.uniform(20, 90)                            # 明るさを足す

    # 局所的なハイライト（丸い光の反射）: 30% の確率
    if random.random() < 0.3:
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        cx, cy = random.uniform(0, arr.shape[1]), random.uniform(0, arr.shape[0])   # 光の中心
        r = random.uniform(0.2, 0.6) * max(arr.shape)                               # 光の広がり
        arr += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * random.uniform(30, 100)

    # センサーノイズ（ザラザラ）
    arr += np.random.normal(0, random.uniform(2, 15), arr.shape)

    # 0〜255 の範囲に収めて整数に戻し、Pillow 画像に戻す
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 小数点や隣セルのドットが角に入り込む: 25% の確率
    if random.random() < 0.25:
        dr = ImageDraw.Draw(img)
        w, h = img.size
        r = random.randint(2, max(3, w // 8))                                  # ドットの半径
        cx = random.choice([random.randint(0, r), random.randint(w - r, w)])  # 左端 or 右端
        cy = random.randint(int(h * 0.55), h)                                 # 下寄り
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=random.randint(150, 255))   # 明るい丸

    # 窓枠の縁が映り込む: 30% の確率
    if random.random() < 0.3:
        dr = ImageDraw.Draw(img)
        w, h = img.size
        side = random.choice(["l", "r", "t", "b"])    # 左・右・上・下のどこか
        t = random.randint(2, max(3, w // 10))         # 縁の太さ
        box = {"l": (0, 0, t, h), "r": (w - t, 0, w, h), "t": (0, 0, w, t), "b": (0, h - t, w, h)}[side]
        dr.rectangle(box, fill=random.randint(0, 255))

    # 最後に CNN の入力サイズへ縮小（LANCZOS = きれいに縮小する方式）
    return img.resize((W, H), Image.LANCZOS)


# ================== メイン処理 ==================
def main():
    # 出力フォルダを 0〜19 ぶん作る（exist_ok=True: 既にあってもエラーにしない）
    for label in range(20):
        os.makedirs(f"{OUT}/{label}", exist_ok=True)

    for label in range(20):
        for i in range(N_PER_CLASS):
            font = random.choice(FONTS)          # フォントをランダムに選ぶ
            if random.random() < 0.8:
                # 80%: 黒地に白文字（日本の水道メーターの主流）
                fg, bg = random.randint(170, 255), random.randint(0, 70)
            else:
                # 20%: 白地に黒文字（赤い桁などの再現）
                fg, bg = random.randint(0, 70), random.randint(170, 255)

            # 窓を作る → 汚す → 保存。f"{i:05d}" は 5 桁ゼロ埋め（00000, 00001, ...）
            augment(sample(label, font, fg, bg)).save(f"{OUT}/{label}/{i:05d}.png")
        print("class", label, "done")


# このファイルを直接実行したときだけ main() を動かす（他のファイルから import されたときは動かさない）
if __name__ == "__main__":
    main()
