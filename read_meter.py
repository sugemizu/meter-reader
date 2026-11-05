"""
=====================================================================
read_meter.py — 水道メーターの写真から数値を読み取る（本体）
=====================================================================

■ 処理の流れ（5 段階）
  1. 検出   : 学習済みの文字検出モデル (PP-OCR) で「文字が並んでいる場所」を全部見つける
  2. 選択   : その中から「数字窓」らしいものを 1 つ選ぶ（横長・黒い背景・大きさ でスコア付け）
  3. 正面化 : 斜めに写った窓を、4 隅を基準に長方形に引き伸ばして正面向きにする
  4. 分割   : 窓を 1 桁ずつのセルに切り分ける
  5. 分類   : 各セルを CNN (digit_cnn.onnx) に入れて数字を判定する

■ 使い方
  python read_meter.py photo.jpg --digits 5 --red 1 --debug
    --digits : 数字窓の桁数（黒＋赤の合計）
    --red    : 右端の赤い桁（小数）の数
    --debug  : 途中経過の画像を出力する

■ ライブラリ
  cv2 (OpenCV)   : 画像処理の定番ライブラリ
  onnxruntime    : ONNX 形式のモデルを動かすエンジン
  rapidocr       : PP-OCR の検出モデルを同梱した軽量 OCR パッケージ
"""

import sys, time, argparse
import numpy as np
import cv2
import onnxruntime as ort

W, H = 28, 40    # CNN に入れる 1 桁画像のサイズ（学習時と同じ）


# ======================================================
# 1. 検出 — 文字がある場所を見つける
# ======================================================
def load_detector():
    """
    PP-OCR の文字検出モデル（学習済み・ONNX）を読み込む。
    rapidocr パッケージがモデルを同梱しているのでダウンロード不要。
    """
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    # 1 回ダミー画像で動かしておく（初回だけ遅いので、本番の計測に含めないため）
    ocr(np.zeros((64, 64, 3), np.uint8), use_det=True, use_cls=False, use_rec=False)
    return ocr


def detect_boxes(ocr, img):
    """
    画像から文字領域を検出して、各領域の 4 隅の座標を返す。
      use_det=True  : 検出だけ使う
      use_cls=False : 文字の向き判定は使わない
      use_rec=False : 文字認識（何と書いてあるか）は使わない。認識は自前の CNN でやる
    戻り値: [4×2 の配列, 4×2 の配列, ...]（領域ごとに 4 点 × (x, y)）
    """
    out = ocr(img, use_det=True, use_cls=False, use_rec=False)
    # 返り値はバージョンにより (結果, 経過時間) のタプルか、結果そのもの
    res = out[0] if isinstance(out, tuple) else out
    if res is None:
        return []
    boxes = []
    for r in res:
        a = np.array(r, dtype=object)
        # 形式 A: [4点座標, 文字, 信頼度] → 先頭が座標
        # 形式 B: 4点座標そのもの（検出のみのときにこの形で返すバージョンがある）
        if a.ndim == 2 and a.shape == (4, 2):
            box = a
        else:
            box = r[0]
        box = np.array(box, dtype=np.float32).reshape(-1, 2)
        if box.shape == (4, 2):
            boxes.append(box)
    return boxes


# ======================================================
# 2. 選択 — 数字窓らしい領域を選ぶ
# ======================================================
def order_pts(pts):
    """
    4 点を「左上, 右上, 右下, 左下」の順に並べ替える。
    （検出器が返す点の順番はバラバラなので、後の射影変換のためにそろえる）
      x+y が最小 → 左上, 最大 → 右下
      y-x が最小 → 右上, 最大 → 左下
    """
    s = pts.sum(1)                          # 各点の x+y
    d = np.diff(pts, axis=1).ravel()        # 各点の y-x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def box_metrics(gray, box):
    """
    検出領域の「幅・高さ・平均の明るさ」を計算する。
    戻り値: (並べ替えた4点, 幅, 高さ, 平均明るさ)
    """
    p = order_pts(box)
    # 幅 = 上辺と下辺の長さの平均、高さ = 左辺と右辺の長さの平均（norm = 2点間の距離）
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    # 領域の内側だけ白く塗ったマスクを作り、その部分の平均明るさを測る
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [p.astype(np.int32)], 255)
    mean = cv2.mean(gray, mask)[0]
    return p, w, h, mean


DIGIT_ASPECT = 0.65   # 1 桁の「幅 ÷ 高さ」のおおよその値（実写 2 枚から: 0.6〜0.71）


def pick_counter_window(gray, boxes, expect_digits):
    """
    検出された領域すべてにスコアを付けて、一番「数字窓っぽい」ものを返す。
    スコアは 3 つの観点の重み付き合計:
      s_dark   : 背景が暗いか（黒地白字の窓は、文字盤の中で飛び抜けて暗い）       … 重み 0.5
      s_aspect : 横長さが「桁数 × DIGIT_ASPECT」の 55%〜110% に入っているか。
                 検出器が白地の赤い桁を取りこぼして枠が短くなることがあるので、
                 ぴったり一致を求めず「範囲内なら満点」にする                    … 重み 0.3
      s_size   : 画像の中で十分大きいか                                        … 重み 0.2
    戻り値: (4点座標, スコア)。候補が無ければ (None, -1)
    """
    best, best_score = None, -1
    for b in boxes:
        p, w, h, mean = box_metrics(gray, b)
        if h < 12 or w < 30:        # 小さすぎる領域は無視
            continue
        aspect = w / h                                       # 縦横比
        target = expect_digits * DIGIT_ASPECT                # 期待する縦横比
        ratio = aspect / target
        if 0.55 <= ratio <= 1.1:
            s_aspect = 1.0                                   # 範囲内なら満点
        else:
            s_aspect = max(0, 1 - abs(ratio - 1))            # 外れるほど減点
        s_dark = (1 - mean / 255) ** 2                       # 暗いほど 1（2 乗で差を強調）
        s_size = min(1, w / (gray.shape[1] * 0.3))           # 画像幅の 30% 以上なら 1
        score = 0.5 * s_dark + 0.3 * s_aspect + 0.2 * s_size
        if score > best_score:                               # 今までで最高なら更新
            best, best_score = p, score
    return best, best_score


def red_mask(img):
    """
    カラー画像から「赤い画素」だけを白にしたマスクを返す。
    HSV 色空間で判定: 色相 H が 0〜10 または 170〜180（赤は両端にまたがる）、
    彩度 S が高く（薄い色を除外）、明度 V がある程度ある（黒を除外）。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= 10) | (h >= 170)) & (s > 100) & (v > 80)).astype(np.uint8) * 255


def extend_to_red_digits(img, p, red_digits):
    """
    文字検出器は「黒地に白文字」の窓は見つけるが、その右隣の
    「白地に赤文字」の小数桁を取りこぼすことが多い。
    そこで赤い領域を探し、枠の右端をそこまで伸ばす。

    条件: 赤い塊のうち、窓と同じ高さにあり、窓の右端から高さの 3 倍以内にあるもの
          （文字盤の下の方にある赤い針は高さが違うので除外される）
      p: 左上, 右上, 右下, 左下 の順の 4 点
    """
    if red_digits <= 0:
        return p
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    y_top, y_bot = min(p[0][1], p[1][1]), max(p[2][1], p[3][1])
    x_right = max(p[1][0], p[2][0])

    m = red_mask(img)
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)   # 赤い塊ごとに位置・大きさを取得
    far_x = None
    for i in range(1, n):                                      # 0 番は背景なので飛ばす
        x, y, w_, h_, area = stats[i]
        cx, cy = cent[i]
        if area < 15:                                          # 小さすぎるノイズは無視
            continue
        if not (y_top - h * 0.2 <= cy <= y_bot + h * 0.2):     # 窓と同じ高さか
            continue
        if not (x_right - h * 0.3 <= cx <= x_right + h * 3):   # 窓の右隣か
            continue
        far_x = x + w_ if far_x is None else max(far_x, x + w_)   # 一番右の赤の右端

    if far_x is None:
        return p                                               # 赤が見つからなければそのまま
    ext = far_x + h * 0.35 - x_right                           # 右端まで + 余白
    if ext <= 0:
        return p
    # 枠は斜めに写っていることもあるので、上辺・下辺の向きに沿って伸ばす
    top_dir = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
    bot_dir = (p[2] - p[3]) / np.linalg.norm(p[2] - p[3])
    q = p.copy()
    q[1] = p[1] + top_dir * ext
    q[2] = p[2] + bot_dir * ext
    return q


# ======================================================
# 3. 正面化 — 斜めの窓を長方形に直す
# ======================================================
def rectify(gray, p, scale_h=64):
    """
    4 隅 p で囲まれた領域を、高さ scale_h の長方形画像に変換する（射影変換）。
    斜めから撮っても、この後の処理は「正面から撮った窓」として扱える。
    """
    w = int((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2)   # 元の幅
    h = int((np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2)   # 元の高さ
    out_w = int(w * scale_h / h)                                               # 縦横比を保った出力幅
    # 変換先の 4 隅（左上, 右上, 右下, 左下）
    dst = np.array([[0, 0], [out_w, 0], [out_w, scale_h], [0, scale_h]], np.float32)
    M = cv2.getPerspectiveTransform(p, dst)          # 変換行列を求める
    return cv2.warpPerspective(gray, M, (out_w, scale_h))   # 変換を適用


# ======================================================
# 4. 分割 — 窓を 1 桁ずつに切る
# ======================================================
def crop_main_band(win, bw):
    """
    上下にのぞく隣の数字を落とすため、「白い画素が多い行」が最も長く続く帯だけを残す。
    （※ 実写では境目がはっきりせず効かないことも多い。その場合は何もしない）
    win : グレー画像, bw : 二値化（白黒）画像
    """
    h = bw.shape[0]
    rows = (bw > 0).mean(1) > 0.04      # 各行で白が 4% 以上あるか（True/False の列）
    best, cur = (0, h), None            # best = 最長の帯 (開始, 終了), cur = 今の帯の開始行
    for i, r in enumerate(list(rows) + [False]):   # 最後に False を足して帯を必ず閉じる
        if r and cur is None:           # 帯の始まり
            cur = i
        elif not r and cur is not None: # 帯の終わり
            if i - cur > best[1] - best[0] or best == (0, h):
                best = (cur, i)         # 今までで最長なら更新
            cur = None
    a, b = best
    m = max(1, int((b - a) * 0.08))     # 上下に 8% の余白
    a, b = max(0, a - m), min(h, b + m)
    if b - a < h * 0.3:                 # 帯が短すぎる＝うまく取れていない → 何もしない
        return win, bw
    return win[a:b], bw[a:b]


def _cut_cell(win, c, cw, invert=False):
    """
    窓画像 win から中心 c・幅 cw のセルを切り出し、CNN の入力サイズに縮小する。
    invert=True なら白黒を反転する（白地に暗い文字 → 暗い背景に明るい文字 にそろえる）
    """
    w = win.shape[1]
    a, b = int(round(c - cw / 2)), int(round(c + cw / 2))   # セルの左端・右端
    pad_l, pad_r = max(0, -a), max(0, b - w)                # 画像からはみ出す量
    cell = win[:, max(0, a):min(w, b)]
    if pad_l or pad_r:                                      # はみ出した分は端の画素を伸ばして埋める
        cell = cv2.copyMakeBorder(cell, 0, 0, pad_l, pad_r, cv2.BORDER_REPLICATE)
    # ---- 白黒の向きをそろえる（常に「暗い背景に明るい数字」にする）----
    # 画像の明るさから自動判定すると、反射で窓全体が明るいときや隣の窓の縁が入ったときに
    # 誤判定しやすい。メーターの構造（黒い桁＝黒地白字、赤い桁＝白地赤字）は分かっているので、
    # 呼び出し側が invert で指定する
    if invert:
        cell = 255 - cell
    return cv2.resize(cell, (W, H), interpolation=cv2.INTER_AREA)   # INTER_AREA = 縮小向き


def split_digits(win, n, win_color=None, red_digits=0, invert_black=False):
    """
    正面化した窓画像 win を n 桁のセルに分割する。

    ・黒い桁（n - red_digits 個）: 「列ごとの白い画素の量」のピーク（＝数字の中心）を探して
      そこにセルの中心を合わせる（等分割だと枠の余白でずれるため）
    ・赤い桁（red_digits 個）: 白地に赤なので明るさでは探せない。
      カラー窓 win_color の赤マスクから、赤い塊の中心をセル中心にする
    戻り値: (セル画像のリスト, 処理後の窓画像)
    """
    h, w = win.shape
    n_black = n - red_digits

    # ---- 赤い桁の中心を先に決める ----
    red_centers = []
    if red_digits > 0 and win_color is not None:
        m = red_mask(win_color)
        proj_r = cv2.GaussianBlur((m > 0).astype(np.float32).sum(0)[None, :], (max(3, int(h * 0.3)) | 1, 1), 0)[0]
        red_centers = _pick_peaks(proj_r, red_digits, min_dist=h * 0.4) or []
    # 赤が見つからなければ、右端を赤い桁とみなす（フォールバック）
    if len(red_centers) < red_digits:
        step = w / n
        red_centers = [w - (red_digits - i - 0.5) * step for i in range(red_digits)]

    # ---- 黒い桁は赤い桁より左側だけで探す ----
    x_limit = int(min(red_centers) - h * 0.3) if red_centers else w
    sub = win[:, :max(1, x_limit)]
    if sub.mean() > 127:                                    # 白地黒字の機種なら反転
        sub = 255 - sub
    _, bw = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Otsu 法で二値化
    _, bw = crop_main_band(sub, bw)
    step = sub.shape[1] / max(1, n_black)                   # 等分割したときの 1 桁の幅
    proj = (bw > 0).astype(np.float32).sum(0)               # 列ごとの白い画素数
    k = max(3, int(step * 0.6)) | 1                         # 平滑化の幅（奇数にする）
    sm = cv2.GaussianBlur(proj[None, :], (k, 1), 0)[0]
    black_centers = _pick_peaks(sm, n_black, min_dist=step * 0.6)
    if black_centers is None:                               # 見つからなければ等分割
        black_centers = [(i + 0.5) * step for i in range(n_black)]
    else:
        # 「0」や「8」は中央がへこむのでピークが端に寄る。
        # ピークを含む「白が連続する区間」の中点を数字の中心にし直す
        black_centers = [_run_center(proj, c, thresh=h * 0.05, max_half=step * 0.5) for c in black_centers]

    # ---- セル幅 = 黒い桁の中心間隔の中央値 ----
    cw = np.median(np.diff(black_centers)) if n_black > 1 else h * DIGIT_ASPECT

    # ---- 赤い桁の中心を「白い箱」で補正 ----
    # 赤マスクは色の薄い部分を取りこぼして中心が偏ることがある。
    # 赤い桁は白い箱の中にあるので、赤の中心から左右に「明るい列」が続く範囲の中点を中心にする
    bright = win.mean(0)                                    # 列ごとの平均明るさ
    # しきい値は「暗い列（下位10%）と明るい列（上位10%）の中間」で自動的に決める
    thr = (np.percentile(bright, 10) + np.percentile(bright, 90)) / 2
    red_centers = [_run_center(bright, c, thresh=thr, max_half=cw * 0.6) for c in red_centers]

    # 黒い桁は黒地白字（invert_black=True の機種なら反転）、赤い桁は白地なので常に反転
    cells = [_cut_cell(win, c, cw, invert=invert_black) for c in black_centers]
    cells += [_cut_cell(win, c, cw, invert=True) for c in red_centers]
    return cells, win


def _run_center(proj, c, thresh, max_half):
    """
    列投影 proj の位置 c から左右に「値が thresh より大きい間」広げて、
    その区間（＝数字のインクがある範囲）の中点を返す。
    広げる幅は片側 max_half まで（隣の数字につながってしまうのを防ぐ）。
    """
    w = len(proj)
    a = b = int(round(c))
    while a - 1 >= 0 and proj[a - 1] > thresh and c - (a - 1) <= max_half:
        a -= 1
    while b + 1 < w and proj[b + 1] > thresh and (b + 1) - c <= max_half:
        b += 1
    return (a + b) / 2


def _pick_peaks(sig, n, min_dist):
    """
    信号 sig から、高い順に n 個のピークを選ぶ。ただし互いに min_dist 以上離れていること。
    戻り値: 左から順に並べたピーク位置。n 個取れなければ None
    """
    idx = np.argsort(-sig)              # 値が大きい順の添字（-sig で降順）
    chosen = []
    for i in idx:
        if sig[i] <= 0:                 # 白が 1 つも無い列まで来たら終了
            break
        # 既に選んだピークのどれとも十分離れていれば採用
        if all(abs(i - c) >= min_dist for c in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return None
    return sorted(chosen)


# ======================================================
# 5. 分類 — 各セルの数字を CNN で判定
# ======================================================
def classify(sess, cells, tta=True, both_polarity=True):
    """
    セル画像を CNN に入れて、各セルの (予測ラベル, 信頼度) を返す。

    tta=True のときは画像を上下左右 2 ピクセルずらした 4 パターンも推論して平均する
    （TTA = Test Time Augmentation。少しのズレに強くなる。追加コストは数 ms）
    both_polarity=True のときは白黒反転した画像でも推論して平均する
    （モデルは両方の向きで学習しているが完全には同じ答えにならないので、平均してブレを減らす）
    """
    # (枚数, 高さ, 幅) → (枚数, 1, 高さ, 幅)、0〜255 → 0〜1
    x = np.stack(cells).astype(np.float32)[:, None] / 255.0
    views = [x]
    if tta:
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            views.append(np.roll(x, (dy, dx), axis=(2, 3)))   # 画像をずらしたコピー
    if both_polarity:
        views += [1 - v for v in views]                        # 全パターンの白黒反転版

    probs = []
    for v in views:
        logits = sess.run(None, {"x": v})[0]                  # CNN を実行 → 20 個のスコア
        # softmax: スコアを「合計 1 の確率」に変換（最大値を引くのは桁あふれ防止）
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
    prob = np.mean(probs, 0)                                  # 5 パターンの平均
    return prob.argmax(1), prob.max(1)      # 最も確率の高いラベル, その確率


def to_reading(labels, probs, red_digits=0):
    """
    桁ごとのラベルを「検針値の文字列」にまとめる。

    ・ラベル 10+d（回転中）は小さい方の d を採用（切り捨て読み）
    ・ある桁が回転中なら、その右側の桁は「…999 → …000」の途中で 9 側にいるはずなので、
      つじつまを合わせるため右側は全部 9 にする
        例: [1, 2→3回転中, 9→0回転中] → "129"
    ・red_digits > 0 なら右から red_digits 桁の前に小数点を入れる

    戻り値: (読み値の文字列, 全桁の中で最低の信頼度, 桁ごとの回転中フラグ)
    """
    digits = []      # 各桁の数値
    rolling = []     # 各桁が回転中かどうか (True/False)
    for l in labels:
        l = int(l)
        rolling.append(l >= 10)
        digits.append(l - 10 if l >= 10 else l)   # 10+d なら d、それ以外はそのまま

    # 最初に回転中だった桁の位置（無ければ None）
    first_roll = next((i for i, r in enumerate(rolling) if r), None)
    if first_roll is not None:
        for j in range(first_roll + 1, len(digits)):   # それより右の桁を 9 に
            digits[j] = 9

    s = "".join(str(d) for d in digits)               # 数字を連結して文字列に
    if red_digits:
        s = s[:-red_digits] + "." + s[-red_digits:]   # 小数点を挿入
    return s, float(np.min(probs)), rolling


# ======================================================
# main — コマンドラインから呼ばれたときの処理
# ======================================================
def main():
    # コマンドライン引数の定義
    ap = argparse.ArgumentParser()
    ap.add_argument("image")                                                   # 画像ファイル
    ap.add_argument("--digits", type=int, default=8, help="数字窓の桁数")
    ap.add_argument("--red", type=int, default=3, help="小数桁（赤字）の数")
    ap.add_argument("--invert", action="store_true", help="黒い桁が白地に黒文字の機種なら指定")
    ap.add_argument("--debug", action="store_true")                            # 付けると True
    a = ap.parse_args()

    img = cv2.imread(a.image)                          # 画像を読み込む（カラー）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # グレースケール版も作る
    ocr = load_detector()                              # 検出モデル
    sess = ort.InferenceSession("digit_cnn.onnx", providers=["CPUExecutionProvider"])   # 認識モデル

    # ---- 1〜5 を順に実行（時間も測る）----
    t0 = time.perf_counter()
    boxes = detect_boxes(ocr, img)                     # 1. 検出
    t1 = time.perf_counter()
    p, score = pick_counter_window(gray, boxes, a.digits)   # 2. 選択
    if p is None:
        print("数字窓が見つかりません"); return
    p = extend_to_red_digits(img, p, a.red)            # 2'. 赤い小数桁が枠外なら右へ広げる
    win = rectify(gray, p)                             # 3. 正面化（グレー）
    win_color = rectify(img, p)                        #    同じ変換をカラーにも（赤い桁を探す用）
    cells, win_n = split_digits(win, a.digits, win_color, a.red, a.invert)   # 4. 分割
    labels, probs = classify(sess, cells)              # 5. 分類
    t2 = time.perf_counter()

    # ---- 結果表示 ----
    reading, conf, rolling = to_reading(labels, probs, a.red)
    print(f"reading: {reading}  min_conf: {conf:.2f}  window_score: {score:.2f}  rolling: {rolling}")
    print(f"detect {1000*(t1-t0):.0f} ms / rest {1000*(t2-t1):.0f} ms")
    print("per-digit:", [(int(l), round(float(c), 2)) for l, c in zip(labels, probs)])

    # ---- デバッグ画像 ----
    if a.debug:
        vis = img.copy()
        cv2.polylines(vis, [p.astype(np.int32)], True, (0, 255, 0), 2)   # 選ばれた窓を緑枠で
        cv2.imwrite("debug_box.jpg", vis)
        cv2.imwrite("debug_window.png", win_n)                             # 正面化した窓
        cv2.imwrite("debug_cells.png", np.hstack(cells))                   # 分割した各桁を横に並べて
        print("debug_*.{jpg,png} を出力しました")


if __name__ == "__main__":
    main()
