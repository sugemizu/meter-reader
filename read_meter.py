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

import os, sys, time, argparse
import numpy as np
import cv2
import onnxruntime as ort

# モデルはスクリプトの場所基準で読む（別ディレクトリから実行しても見つかるように）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_cnn.onnx")
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
    縦長の検出枠は 90° 回転した写真の窓として、点を回して横長として評価する。
    戻り値: (4点座標, スコア)。候補が無ければ (None, -1)
    """
    best, best_score = None, -1
    for b in boxes:
        p, w, h, mean = box_metrics(gray, b)
        img_w = gray.shape[1]                                # s_size の基準にする「窓から見た画像の幅」
        if h > w:
            # 縦長の枠は 90° 回った窓とみなす。order_pts は画像座標で並べるので縦長では p0→p1 が短い辺になる。
            # 点を 1 つ回して p0→p1 を文字の並ぶ向きにそろえる（90° と 270° の区別は locate_and_read の 0°/180° 判定で付く）
            p = np.roll(p, -1, axis=0)
            w, h = h, w
            img_w = gray.shape[0]
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
        s_size = min(1, w / (img_w * 0.3))                   # 画像幅の 30% 以上なら 1
        score = 0.5 * s_dark + 0.3 * s_aspect + 0.2 * s_size
        if score > best_score:                               # 今までで最高なら更新
            best, best_score = p, score
    return best, best_score


RED_HUE_LO, RED_HUE_HI = 12, 165   # 赤の色相範囲（OpenCV の H は 0〜180）。従来の 10/170 より少し広い（露出・ガンマで色相が動くため）


def red_mask(img, roi_mask=None, return_stats=False, min_diff=15):
    """
    カラー画像から「赤い画素」だけを白にしたマスクを返す。
    しきい値は画像（roi_mask があればその内側）の中で相対的に決める:
      score = R - max(G, B)（0〜255）
      noise = score の 90 パーセンタイル（ほとんどの画素は赤くないので、これが雑音の水準）
      peak  = score の 99.5 パーセンタイル（赤い桁の画素はおおむね 0.5% 以上ある）
      peak - noise < min_diff（既定 15）なら「赤は無い」として空のマスク（赤の無い窓で雑音を拾わないための下限）。
      以前は peak < 3*noise + 4 だったが、探索範囲に文字盤が入ると noise 自体が高く（素の meter4 で 17）、
      暖色かぶり +10 で赤なしと誤判定したので、倍率でなく差で見る（赤い桁の peak - noise は素で 50、かぶり +10〜+20 でも 55 前後）
      thr = noise + 0.4*(peak - noise)、 mask = (score > thr) & 色相が赤
    HSV の彩度・明度の絶対しきい値（S>100, V>80）は、低コントラスト・ぼけ・露出不足で赤が全部消えたので廃止。
    Otsu も試したが、赤が数 % しかなくヒストグラムが片寄り、しきい値が雑音のすぐ上に張り付いたので不採用。
    return_stats=True なら (mask, dict(thr, peak, noise)) を返す（--debug 用）
    """
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    score = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    vals = score[roi_mask > 0] if roi_mask is not None else score.ravel()
    empty = np.zeros(score.shape, np.uint8)
    if vals.size == 0:
        return (empty, dict(thr=None, peak=0.0, noise=0.0)) if return_stats else empty
    noise = float(np.percentile(vals, 90))
    peak = float(np.percentile(vals, 99.5))
    if peak - noise < min_diff:
        return (empty, dict(thr=None, peak=peak, noise=noise)) if return_stats else empty
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0]       # 赤なしの早期 return では不要なのでここで変換
    thr = noise + 0.4 * (peak - noise)       # 0.4: 赤セルの値と雑音の間の雑音寄り。0.5 だと低コントラストで細い線が切れる
    m = ((score > thr) & ((hue <= RED_HUE_LO) | (hue >= RED_HUE_HI))).astype(np.uint8) * 255
    if roi_mask is not None:
        m &= roi_mask
    return (m, dict(thr=thr, peak=peak, noise=noise)) if return_stats else m


def extend_to_red_digits(img, p, red_digits):
    """
    文字検出器は「黒地に白文字」の窓は見つけるが、その右隣の
    「白地に赤文字」の小数桁を取りこぼすことが多い。
    そこで赤い領域を探し、枠の右端をそこまで伸ばす。

    窓座標系（窓を正面化した座標）で探す。画像座標で「同じ高さ・右隣」と判定すると、
    窓が 30° 以上傾いたり天地が逆（赤が画像の左）だったりすると探せない。
    また画像全体を HSV 変換せず窓まわりだけ処理するので速い。

    条件: 赤い塊のうち、窓と同じ高さ（上下 0.2h まで）にあり、窓の右端から高さの 3 倍以内にあるもの
          （文字盤の下の方にある赤い針は高さが違うので除外される）
      p: 左上, 右上, 右下, 左下 の順の 4 点
    """
    if red_digits <= 0:
        return p
    unit, vpad, reach = 64, 0.2, 3.0                           # 窓の高さを 64 行にして、上下 0.2h・右へ 3h を見る
    vl, vr = p[3] - p[0], p[2] - p[1]                          # 左辺・右辺の「下向き」ベクトル
    h = float(np.linalg.norm(vl + vr)) / 2
    top_dir = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
    bot_dir = (p[2] - p[3]) / np.linalg.norm(p[2] - p[3])
    # 探索枠: 右辺を各辺の向きに 3h 伸ばし、上下を 0.2h ずつ広げる
    big = p.copy()
    big[1] = p[1] + top_dir * reach * h
    big[2] = p[2] + bot_dir * reach * h
    big[0] -= vpad * vl                                        # 上下は左右それぞれの辺のベクトルで広げる（台形の枠で辺がずれない）
    big[1] -= vpad * vr
    big[2] += vpad * vr
    big[3] += vpad * vl
    sh = int(round(unit * (1 + 2 * vpad)))
    M, ow = _quad_homography(big, sh)
    wc = cv2.warpPerspective(img, M, (ow, sh))
    pr = cv2.perspectiveTransform(np.asarray(p, np.float32)[None], M)[0]
    x_right = float(max(pr[1, 0], pr[2, 0]))                   # 元の枠の右端（窓座標）

    m = red_mask(wc)                                           # 探索枠の全体が範囲なので roi_mask は不要
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)    # 赤い塊ごとに位置・大きさを取得
    amin = max(15, 0.002 * unit * unit)                        # 小さすぎるノイズは無視
    blobs = []
    for i in range(1, n):                                      # 0 番は背景なので飛ばす
        x, y, w_, h_, area = stats[i]
        cx, cy = cent[i]
        if area < amin:
            continue
        # 帯を縦に横切る赤い針は、探索範囲の上下端に届くか、数字より背が高い。
        # 重心だけで判定すると針の重心が帯の中に入って拡張が針まで伸びるので、こうした塊は捨てる
        if y == 0 or y + h_ >= sh or h_ > 1.2 * unit:
            continue
        blobs.append((cx, x + w_, area))
    # 赤なし判定を差で見るようにして雑音に寛容になったぶん、探索範囲で最大の塊の 1/10 未満の小さな塊
    # （窓の右の文字盤の微小な赤み。面積 15〜17 で拡張を 2 倍近くに伸ばした例があった）は雑音として捨てる。
    # 最大の塊は窓の中の赤い桁のこともあるので、位置で絞る前の全部から求める
    amax = max((a for _, _, a in blobs), default=0)
    far_x = None
    for cx, right, area in blobs:
        if area < 0.1 * amax:
            continue
        if not (x_right - unit * 0.3 <= cx <= x_right + unit * reach):   # 窓の右隣か
            continue
        far_x = right if far_x is None else max(far_x, right)               # 一番右の赤の右端

    if far_x is None:
        return p                                               # 赤が見つからなければそのまま
    ext = (far_x + unit * 0.35 - x_right) * h / unit           # 右端まで + 余白（窓座標 → 画像の長さ）
    if ext <= 0:
        return p
    # 枠は斜めに写っていることもあるので、上辺・下辺の向きに沿って伸ばす
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


def _quad_homography(p, scale_h):
    """
    4 隅 p を、高さ scale_h の長方形に写す射影行列と出力幅を返す（rectify と同じ考え方）。
    refine_vertical と extend_to_red_digits が「窓座標系」で処理するための共通部品。
    """
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    out_w = max(8, int(w * scale_h / h))                      # 極端に細い枠でも幅 0 にしない
    dst = np.array([[0, 0], [out_w, 0], [out_w, scale_h], [0, scale_h]], np.float32)
    return cv2.getPerspectiveTransform(np.asarray(p, np.float32), dst), out_w


def refine_vertical(gray, p, n, red_digits, invert_black=False,
                    expand=0.5, unit=32, rel=0.1, face=0.7, margin=0.1, dbg=None):
    """
    検出枠 p（左上, 右上, 右下, 左下）の上下の辺を、数字が並ぶ帯の上下縁に合わせ直す。
    左右の辺（の直線）はそのまま。上下方向だけを窓座標系で平行移動・伸縮する。
    検出枠の上下が数 px ずれると、上の桁の下端などがセルに入って 4→1 などに化けていたため。

    ・窓の暗さで縁を探す案は不採用: メーターの文字盤も黒いので、窓の内外で暗さが区別できなかった。
      そこで「黒桁の列の、白インクの行ごとの割合」を使う（窓の上下の縁にはインク 0 の行がある）
    ・白い文字盤の機種では窓の外が「ほぼ全部明るい行」になるので、割合が face 以上の行でも止める
    ・傾きは補正しない（水平のまま）。左右で帯を求めて直線フィットする版は悪化したため

    dbg: dict を渡すと --debug 用に正面化画像・行プロファイル・各境界の行を書き込む
    戻り値: (新しい 4 点 float32, (a, b) 帯の上端・下端の行)  または失敗時 (p, None)
    """
    vl, vr = p[3] - p[0], p[2] - p[1]                        # 左辺・右辺の「下向き」ベクトル（台形の枠で辺がずれないよう別々に広げる）
    big = p.copy()
    big[0] -= expand * vl                                    # 上下を高さの expand 倍ずつ広げた枠
    big[1] -= expand * vr
    big[2] += expand * vr
    big[3] += expand * vl
    sh = int(round(unit * (1 + 2 * expand)))
    M, ow = _quad_homography(big, sh)
    w = cv2.warpPerspective(gray, M, (ow, sh))
    if invert_black:                                         # 白地黒字の機種は反転して「明るい＝インク」にそろえる
        w = 255 - w
    y0, y1 = int(round(unit * expand)), int(round(unit * (1 + expand)))   # 元の窓の行範囲
    # 黒桁の列だけを見る。赤い桁の白い箱は黒い帯より上下に張り出す機種があるので入れない
    x0, x1 = int(ow * 0.02), int(ow * (n - red_digits) / n * 0.97)
    sub = w[:, x0:max(x0 + 1, x1)]                           # 幅 0 を防ぐ
    t, _ = cv2.threshold(sub[y0:y1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # 元の窓の中で Otsu
    prof = (sub > t).mean(1).astype(np.float32)              # 行ごとの白インクの割合
    prof = cv2.GaussianBlur(prof[:, None], (1, 3), 0)[:, 0]  # 縦 3 タップ。強くすると縁の隙間が埋まる
    c0, c1 = y0 + unit // 4, y1 - unit // 4                  # 元の窓の中央半分から種を選ぶ
    seed = c0 + int(np.argmax(prof[c0:c1]))
    thr = max(0.01, rel * prof[c0:c1].max())  # 帯の中のくぼみ（最大の 12〜15%）は通し、縁の 0 で止める。
    # 下限 0.01 は max が 0 に近い（インクがほぼ無い）ときも thr > 0 を保ち、インク 0 の行で必ず止めるため
    lo, hi = max(0, y0 - unit // 4), min(sh, y1 + unit // 4)   # 広げる範囲は元の窓の上下 1/4 まで
    a = b = seed
    while a - 1 >= lo and thr < prof[a - 1] < face:
        a -= 1
    while b + 1 < hi and thr < prof[b + 1] < face:
        b += 1
    rh = b - a + 1
    if dbg is not None:
        dbg.update(img=w, prof=prof, y0=y0, y1=y1, a=a, b=b, thr=thr, face=face, lo=lo, hi=hi, x0=x0, x1=x1)
    # 帯が外のラベルとつながった（長すぎる）、数字が見えていない（短すぎる）、探索範囲の端に届いた → 失敗
    if not (0.5 * unit <= rh <= 1.15 * unit) or a <= lo or b >= hi - 1:
        return p, None
    m = margin * rh                                          # 上下に 10% の余白（meter4 で正解枠に一番近かった値）
    rect = np.float32([[0, a - m], [ow, a - m], [ow, b + 1 + m], [0, b + 1 + m]])
    q = cv2.perspectiveTransform(rect[None], np.linalg.inv(M))[0].astype(np.float32)   # 元画像の座標へ戻す
    return q, (a, b)


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
    if b <= a:                                              # 幅 0 だと空セルになり cv2.resize が落ちるので最低 1 画素にする
        b = a + 1
    # 窓の外に完全にはみ出すと切り出しが空配列になり、copyMakeBorder が不定の値を返す
    # （ピッチ外挿のセルが窓の右にはみ出すと隣のメモリの値が混ざった）。左端を窓内に収めて端の 1 列から複製させる
    a = min(a, w - 1)
    b = max(b, a + 1, 1)
    a = min(a, b - 1)
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


def split_digits(win, n, win_color=None, red_digits=0, invert_black=False,
                 win_red=None, info=None):
    """
    正面化した窓画像 win を n 桁のセルに分割する。

    ・黒い桁（n - red_digits 個）: 「列ごとの白い画素の量」のピーク（＝数字の中心）を探して
      そこにセルの中心を合わせる（等分割だと枠の余白でずれるため）
    ・赤い桁（red_digits 個）: 白地に赤なので明るさでは探せない。
      カラー窓 win_color の赤マスクから、赤い塊の中心をセル中心にする

    win_red: 赤い桁のセルを切り出すグレー窓。None なら win から切る。
      refine_vertical で枠を直したときは、精緻化前の枠で正面化した窓を渡す。黒桁は精緻化で良くなるが、
      回転中の赤桁（2→3）は切り出しのわずかな違いで CNN が 12 と 6 の間を揺れるため、赤桁の縦範囲は変えない
      （精緻化後の窓から切ると strict が 0.21 → 0.10 に落ちた）
    info: dict を渡すと black_centers / red_centers / cw / red_fallback を書き込む（bench と --debug 用）
    戻り値: (セル画像のリスト, 処理後の窓画像)
    """
    h, w = win.shape
    n_black = n - red_digits

    # ---- 赤い桁の中心を先に決める ----
    red_centers = []
    if red_digits > 0 and win_color is not None:
        m = red_mask(win_color, min_diff=10)             # 窓の中だけを見るので雑音が少なく、低コントラストの赤い桁（peak 12 程度）も拾う
        proj_r = cv2.GaussianBlur((m > 0).astype(np.float32).sum(0)[None, :], (max(3, int(h * 0.3)) | 1, 1), 0)[0]
        # 赤い桁は黒い桁より右側にあるはず。天地逆の写真や窓の左に赤い線・枠があると、
        # 左の赤を赤い桁と誤認して黒い桁の領域が潰れ、空セルになって落ちていたので、
        # 黒い桁の領域（の手前）より左の赤は無視して、赤の探索を窓の右側に限定する
        # 余白は h*0.8。斜めの写真で窓の右に余白が付くと赤い桁が左に寄るため、
        # 黒い桁の領域の境界からやや左まで許す（赤が窓の左端 x≈0.1w にある元の落ちる例は依然として除外される）
        proj_r[:max(0, int(w * n_black / n - h * 0.8))] = 0
        red_centers = _pick_peaks(proj_r, red_digits, min_dist=h * 0.4) or []
    red_fallback = len(red_centers) < red_digits
    if red_fallback:
        # 赤が見つからなければ、黒い桁のピッチから赤い桁の位置を外挿する。
        # 以前は窓の右端を赤い桁とみなしていたが、枠が「▶」や最後の黒桁まで含むと取り違えて全桁が 1 つずれた。
        # x_limit は等分割の境界から半セル右まで（枠が黒桁だけでも最後の黒桁の中心は入り、
        # 「▶」まで含んでも赤い箱の中心までは届かない）
        x_limit = int(w * (n_black + 0.5) / n)
        # 赤が検出できなくても白い箱が枠に含まれることがある（赤だけ色が薄い・箱の中が空など）。
        # 箱の左半分の明るい列を黒桁のピークとして拾うと黒桁が 1 桁ずれる（meter4 の赤桁を白で塗ると 0411.5）ので、
        # 右側の明るい連続区間（白い箱）を探し、その左端の少し手前で打ち切る
        box_left = _white_box_left(win, n_black, n, h)
        if box_left is not None:
            x_limit = min(x_limit, int(box_left - 0.1 * h))
        black_centers = _black_centers(win, x_limit, n_black, h)
        pitch = np.median(np.diff(black_centers)) if n_black > 1 else h * DIGIT_ASPECT
        red_centers = [black_centers[-1] + pitch * (i + 1) for i in range(red_digits)]
    else:
        # ---- 黒い桁は赤い桁より左側だけで探す ----
        x_limit = int(min(red_centers) - h * 0.3) if red_centers else w
        # 万一赤の位置が左に寄っても黒い桁の領域が潰れないよう、等分割の半分は確保する（保険）
        x_limit = max(x_limit, int(w * n_black / n * 0.5))
        black_centers = _black_centers(win, x_limit, n_black, h)

    # ---- セル幅 = 黒い桁の中心間隔の中央値 ----
    cw = np.median(np.diff(black_centers)) if n_black > 1 else h * DIGIT_ASPECT
    cw = max(cw, 2.0)                                       # 幅が 1 画素未満だと空セルになるので下限を付ける（保険）

    # ---- 赤い桁の中心を「白い箱」で補正 ----
    # 赤マスクは色の薄い部分を取りこぼして中心が偏ることがある。
    # 赤い桁は白い箱の中にあるので、赤の中心から左右に「明るい列」が続く範囲の中点を中心にする
    bright = win.mean(0)                                    # 列ごとの平均明るさ
    # しきい値は「暗い列（下位10%）と明るい列（上位10%）の中間」で自動的に決める
    thr = (np.percentile(bright, 10) + np.percentile(bright, 90)) / 2
    red_centers = [_run_center(bright, c, thresh=thr, max_half=cw * 0.6) if 0 <= c < w else c for c in red_centers]

    # 黒い桁は黒地白字（invert_black=True の機種なら反転）、赤い桁は白地なので常に反転
    cells = [_cut_cell(win, c, cw, invert=invert_black) for c in black_centers]
    if win_red is not None:
        # 精緻化前の窓から切る。2 つの窓は左右の辺が同じ直線上なので、x は幅の比で対応する
        sc = win_red.shape[1] / w
        cells += [_cut_cell(win_red, c * sc, cw * sc, invert=True) for c in red_centers]
    else:
        cells += [_cut_cell(win, c, cw, invert=True) for c in red_centers]
    if info is not None:
        info.update(black_centers=list(black_centers), red_centers=list(red_centers),
                    cw=float(cw), red_fallback=red_fallback)
    return cells, win


def _white_box_left(win, n_black, n, h):
    """
    窓の右側にある「明るい列が続く区間」（赤い桁の白い箱）の左端の x を返す。無ければ None。
    列の平均明るさを「暗い列（下位 10%）と明るい列（上位 10%）の中間」で二値化し、右から見て
    幅 h*0.4 以上の最初の区間を箱とみなす（黒桁の白い線は幅が狭く、列平均もしきい値に届かない）。
    箱は黒い桁の領域より右にあるはずなので、黒い桁の領域の境界から h*0.8 手前より左で始まる区間は
    箱ではなく黒桁の一部とみなして無視する（赤の探索と同じ余白）
    """
    w = win.shape[1]
    bright = win.mean(0)
    thr = (np.percentile(bright, 10) + np.percentile(bright, 90)) / 2
    on = bright >= thr
    lowest = w * n_black / n - h * 0.8
    i = w - 1
    while i >= 0:
        if not on[i]:
            i -= 1
            continue
        j = i
        while j - 1 >= 0 and on[j - 1]:
            j -= 1
        if i - j + 1 >= h * 0.4 and j >= lowest:
            return float(j)
        i = j - 1
    return None


def _black_centers(win, x_limit, n_black, h):
    """
    win[:, :x_limit] で黒い桁の中心を n_black 個探す（列投影のピーク → 白が連続する区間の中点）。
    戻り値: 中心のリスト。ピークが取れなければ等分割の中心
    """
    sub = win[:, :max(1, x_limit)]
    if sub.mean() > 127:                                    # 白地黒字の機種なら反転
        sub = 255 - sub
    _, bw = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Otsu 法で二値化
    _, bw = crop_main_band(sub, bw)
    step = sub.shape[1] / max(1, n_black)                   # 等分割したときの 1 桁の幅
    proj = (bw > 0).astype(np.float32).sum(0)               # 列ごとの白い画素数
    k = max(3, int(step * 0.6)) | 1                         # 平滑化の幅（奇数にする）
    sm = cv2.GaussianBlur(proj[None, :], (k, 1), 0)[0]
    peaks = _pick_peaks(sm, n_black, min_dist=step * 0.6)
    if peaks is None:                                       # 見つからなければ等分割
        return [(i + 0.5) * step for i in range(n_black)]
    # 「0」や「8」は中央がへこむのでピークが端に寄る。
    # ピークを含む「白が連続する区間」の中点を数字の中心にし直す
    return [_run_center(proj, c, thresh=h * 0.05, max_half=step * 0.5) for c in peaks]


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
    if n <= 0:                          # 0 個以下を求められたら何も選ばない
        return []
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
    if len(cells) == 0:
        raise ValueError("セルが切り出せませんでした")
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
# 選択〜分類をまとめる（main と bench から呼ぶ）
# ======================================================
def red_side_score(img, p, n):
    """
    枠 p を正面化したカラー窓で、右端 1 桁ぶん（幅 W/n）と左端 1 桁ぶんの「白い箱の中の赤」の画素割合の差を返す。
    正なら赤が右（＝正しい向き）。
    「白い箱の中」の判定: 赤マスクを 7x7 で膨らませた輪（赤自身を除く）の明るさの中央値が、
    窓の列平均の (10% 点 + 90% 点)/2 より明るい。黒地の上の赤い点や線（窓の左に赤い物を置いたテスト）を
    赤い桁と取り違えないため。この条件が無いと 4 件で誤って 180° を選んだ
    """
    wc = rectify(img, p)
    g = cv2.cvtColor(wc, cv2.COLOR_BGR2GRAY)
    m8 = red_mask(wc)
    m = m8 > 0
    W = m.shape[1]
    k = max(1, int(W / n))                                   # 1 桁ぶんの幅
    ring = (cv2.dilate(m8, np.ones((7, 7), np.uint8)) > 0) & ~m
    col = g.mean(0)
    thr = (np.percentile(col, 10) + np.percentile(col, 90)) / 2

    def amount(sl):
        r = ring[:, sl]
        if not r.any():
            return 0.0
        return float(m[:, sl].mean()) if np.median(g[:, sl][r]) > thr else 0.0   # 白い箱の中でなければ数えない
    return amount(slice(W - k, W)) - amount(slice(0, k))


def locate_and_read(img, gray, boxes, n, red_digits, invert_black, sess, debug=None):
    """
    2.選択 〜 5.分類 をまとめて行い、向き（0°/180°）も決める。main と bench から呼ぶ。
    枠を 180° 回した候補（点の並びを 2 つ回す）も同じ処理に通して、次の規則で選ぶ:
      ・赤い桁がある機種: 「白い箱の中の赤」が右端にある方（red_side_score の差が 0.03 より大きいとき）。
        赤は必ず右端の白い箱にあるので強い手がかり。正しい向きの右端 1 桁の赤は 0.10〜0.15、逆向きでは 0。
        しきい値 0.03 は、小さい画像の赤の誤検出（0.014）に反応しない余裕
      ・それ以外: 黒桁の平均信頼度が 0°より 0.2 以上高いときだけ 180°。
        CNN は逆さの 0 や 1 も自信を持って読むので、信頼度だけだと正しい向きなのに 27% で逆を選んだ
    戻り値は dict（数字窓が無ければ None）:
      labels, probs, score, rot, refined, red_fallback,
      p_pick（選択直後）, p_ext（向き補正＋赤拡張後・精緻化前）, p_win（最終の枠）,
      win, win_color, cells, centers（黒＋赤のセル中心。win の x 座標）, cand（両候補の conf_black と rs。例外で落ちた候補は rot と err のみ）
    """
    p, score = pick_counter_window(gray, boxes, n)       # 2. 選択（縦長は回してある）
    if p is None:
        return None
    results, errs, last_err = [], [], None
    for k, c in enumerate([p, np.roll(p, 2, axis=0)]):   # 0° と 180°（点の並びを 2 つ回す）
        try:
            e = extend_to_red_digits(img, c, red_digits)     # 2'. 赤い小数桁が枠外なら右へ広げる
            dbg = {} if debug else None
            q, rows = refine_vertical(gray, e, n, red_digits, invert_black, dbg=dbg)   # 2''. 上下の辺を帯に合わせる
            win, win_color = rectify(gray, q), rectify(img, q)
            win_red = rectify(gray, e) if rows is not None else None   # 赤セルは精緻化前の枠から切る
            info = {}
            cells, _ = split_digits(win, n, win_color, red_digits, invert_black, win_red=win_red, info=info)
            labels, probs = classify(sess, cells)
            rs = red_side_score(img, e, n) if red_digits > 0 else 0.0
        except (cv2.error, ValueError, IndexError, ZeroDivisionError) as ex:
            # 窓が壊れる・空配列になるなど「その向きでは読めない」失敗だけを候補から外す。
            # それ以外（バグ）まで握りつぶすと、0° だけ例外で落ちたときに 180° が黙って採用される。
            # 失敗した候補も cand に err として残し、main が表示する
            errs.append(dict(rot=180 * k, err=f"{type(ex).__name__}: {ex}"))
            last_err = ex
            continue
        results.append(dict(labels=labels, probs=probs, score=score, rot=180 * k, refined=rows is not None,
                            red_fallback=info["red_fallback"], p_pick=p, p_ext=e, p_win=q, rows=rows,
                            win=win, win_color=win_color, win_red=win_red, cells=cells,
                            centers=info["black_centers"] + info["red_centers"], cw=info["cw"], refine_dbg=dbg,
                            conf_black=float(np.mean(probs[:n - red_digits])), rs=float(rs)))
    if not results:
        raise last_err
    best = results[0]
    if len(results) == 2:
        a, b = results
        if red_digits > 0 and abs(a["rs"] - b["rs"]) > 0.03:
            best = a if a["rs"] >= b["rs"] else b
        elif b["conf_black"] > a["conf_black"] + 0.2:
            best = b
    best = dict(best)
    best["cand"] = [dict(rot=r["rot"], conf_black=r["conf_black"], rs=r["rs"]) for r in results] + errs
    return best


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
    if a.digits < 1:
        ap.error("--digits は 1 以上にしてください")
    if not 0 <= a.red < a.digits:                      # 赤い桁は 0 以上、全桁数より少なくする
        ap.error("--red は 0 以上 --digits 未満にしてください")

    # cv2.imread は Windows で日本語などの非 ASCII パスを読めないので、
    # バイト列として読んでから imdecode する。読めなければ明示的なメッセージで終了する
    try:
        img = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)   # 画像を読み込む（カラー）
    except (OSError, ValueError):
        img = None
    if img is None:
        print(f"画像を読み込めません: {a.image}", file=sys.stderr)
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # グレースケール版も作る
    ocr = load_detector()                              # 検出モデル
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])   # 認識モデル

    # ---- 1〜5 を順に実行（時間も測る）----
    t0 = time.perf_counter()
    boxes = detect_boxes(ocr, img)                     # 1. 検出
    t1 = time.perf_counter()
    res = locate_and_read(img, gray, boxes, a.digits, a.red, a.invert, sess, debug=a.debug)   # 2〜5
    if res is None:
        print("数字窓が見つかりません"); return
    labels, probs, score = res["labels"], res["probs"], res["score"]
    t2 = time.perf_counter()

    # ---- 結果表示 ----
    reading, conf, rolling = to_reading(labels, probs, a.red)
    print(f"reading: {reading}  min_conf: {conf:.2f}  window_score: {score:.2f}  rolling: {rolling}")
    print(f"window: rot={res['rot']} refined={res['refined']} red_fallback={res['red_fallback']}")
    for c in res["cand"]:
        if "err" in c:                                  # 片方の向きが例外で落ちたことを黙って流さない
            print(f"warning: rot={c['rot']} の候補が失敗しました: {c['err']}")
    print(f"detect {1000*(t1-t0):.0f} ms / rest {1000*(t2-t1):.0f} ms")
    print("per-digit:", [(int(l), round(float(c), 2)) for l, c in zip(labels, probs)])

    # ---- デバッグ画像 ----
    if a.debug:
        vis = img.copy()
        cv2.polylines(vis, [res["p_pick"].astype(np.int32)], True, (255, 0, 0), 1)   # 青: 検出器が出した枠（選択直後）
        cv2.polylines(vis, [res["p_ext"].astype(np.int32)], True, (0, 255, 255), 1)  # 黄: 向き補正＋赤拡張後（精緻化前）
        cv2.polylines(vis, [res["p_win"].astype(np.int32)], True, (0, 255, 0), 2)    # 緑: 最終の枠（精緻化後）
        x0, y0 = res["p_win"][0]
        cv2.circle(vis, (int(x0), int(y0)), 6, (0, 0, 255), -1)                      # 赤丸: 窓の左上（向きが分かる）
        rows = res["rows"]
        txt = f"refine: ok a={rows[0]} b={rows[1]}" if rows else "refine: fallback"
        fs = max(0.6, img.shape[1] / 1500)
        cv2.putText(vis, txt, (10, int(30 * fs) + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2)
        cv2.putText(vis, f"rot={res['rot']}", (10, int(65 * fs) + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2)
        cv2.imwrite("debug_box.jpg", vis)
        cv2.imwrite("debug_window.png", res["win"])                        # 正面化した窓
        cv2.imwrite("debug_cells.png", np.hstack(res["cells"]))            # 分割した各桁を横に並べて
        # 精緻化の中間画像: 縦に広げた正面化画像（左）と行プロファイルの棒グラフ（右）
        d = res["refine_dbg"]
        if d:
            sh = d["img"].shape[0]
            left = cv2.cvtColor(d["img"], cv2.COLOR_GRAY2BGR)
            bars = np.full((sh, 100, 3), 255, np.uint8)
            for y in range(sh):
                cv2.line(bars, (0, y), (int(d["prof"][y] * 99), y), (80, 80, 80), 1)
            for xx in (d["thr"], d["face"]):                               # 灰: rel*max と face のしきい値
                cv2.line(bars, (int(xx * 99), 0), (int(xx * 99), sh - 1), (160, 160, 160), 1)
            for y in (d["y0"], d["y1"]):                                   # 青: 元の窓の範囲
                cv2.line(left, (0, y), (left.shape[1] - 1, y), (255, 0, 0), 1)
                cv2.line(bars, (0, y), (99, y), (255, 0, 0), 1)
            if rows:
                for y in (rows[0], rows[1] + 1):                           # 緑: 採用した帯
                    cv2.line(left, (0, y), (left.shape[1] - 1, y), (0, 255, 0), 1)
                    cv2.line(bars, (0, y), (99, y), (0, 255, 0), 1)
            cv2.imwrite("debug_refine.png", np.hstack([left, bars]))
        # 赤マスク: 正面化したカラー窓の下に赤マスクを並べる
        m, st = red_mask(res["win_color"], return_stats=True)
        cv2.imwrite("debug_redmask.png", np.vstack([res["win_color"], cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)]))
        print("refine: " + (f"ok rows=({rows[0]},{rows[1]})" if rows else "fallback"))
        if a.red > 0:
            if res["red_fallback"]:
                print(f"red: fallback(pitch={res['cw']:.1f})")
            else:
                thr = "None" if st["thr"] is None else f"{st['thr']:.1f}"
                print(f"red: thr={thr} peak={st['peak']:.1f} noise={st['noise']:.1f} "
                      f"found={[round(c) for c in res['centers'][a.digits - a.red:]]}")
        c = {r["rot"]: r for r in res["cand"]}
        f = lambda r, k: "-" if r not in c or "err" in c[r] else f"{c[r][k]:.2f}"
        print(f"orient: rot={res['rot']} conf0={f(0, 'conf_black')} conf180={f(180, 'conf_black')} "
              f"red0={f(0, 'rs')} red180={f(180, 'rs')}")
        print("debug_*.{jpg,png} を出力しました")


if __name__ == "__main__":
    main()
