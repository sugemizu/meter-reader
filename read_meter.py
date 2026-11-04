"""
水道メーター読み取り PoC
  1. PP-OCR の DB 検出器（学習済み, rapidocr 同梱 ONNX）でテキスト領域を検出
  2. 数字窓らしい領域をフィルタ（横長・暗背景など）
  3. 4点で射影変換して正面化
  4. 列方向の投影で桁境界を推定して等分割
  5. 合成データ学習済み CNN (ONNX) で 1 桁ずつ分類
使い方: python read_meter.py photo.jpg [--digits 8] [--debug]
"""
import sys, time, argparse
import numpy as np
import cv2
import onnxruntime as ort

W, H = 28, 40

# ---------- 1. 検出 ----------
def load_detector():
    # PP-OCR の DB 検出モデル (ONNX) を同梱している rapidocr を使う。
    # ネットワーク不要・ONNX Runtime のみで動くので端末移植もしやすい。
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    ocr(np.zeros((64, 64, 3), np.uint8), use_det=True, use_cls=False, use_rec=False)  # ウォームアップ
    return ocr

def detect_boxes(ocr, img):
    res, _ = ocr(img, use_det=True, use_cls=False, use_rec=False)
    return [np.array(r[0], dtype=np.float32) for r in (res or [])]   # 各 4x2

# ---------- 2. フィルタ ----------
def order_pts(pts):
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)

def box_metrics(gray, box):
    p = order_pts(box)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [p.astype(np.int32)], 255)
    mean = cv2.mean(gray, mask)[0]
    return p, w, h, mean

def pick_counter_window(gray, boxes, expect_digits):
    """数字窓候補をスコア付けして最良を返す"""
    best, best_score = None, -1
    for b in boxes:
        p, w, h, mean = box_metrics(gray, b)
        if h < 12 or w < 30:
            continue
        aspect = w / h
        # 期待アスペクト比: 桁数 × 約0.7
        target = expect_digits * 0.7
        s_aspect = max(0, 1 - abs(aspect - target) / target)
        s_dark = 1 - mean / 255          # 黒地白字を優先
        s_size = min(1, w / (gray.shape[1] * 0.3))
        score = 0.5 * s_aspect + 0.3 * s_dark + 0.2 * s_size
        if score > best_score:
            best, best_score = p, score
    return best, best_score

# ---------- 3. 正面化 ----------
def rectify(gray, p, scale_h=64):
    w = int((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2)
    h = int((np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2)
    out_w = int(w * scale_h / h)
    dst = np.array([[0, 0], [out_w, 0], [out_w, scale_h], [0, scale_h]], np.float32)
    M = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(gray, M, (out_w, scale_h))

# ---------- 4. 分割 ----------
def crop_main_band(win, bw):
    """
    行方向の投影から、最も太い「白い行の帯」＝主数字の帯だけを残す。
    上下にのぞく隣の数字の頭/足を落とすのが目的。
    """
    h = bw.shape[0]
    rows = (bw > 0).mean(1) > 0.04
    best, cur = (0, h), None
    for i, r in enumerate(list(rows) + [False]):
        if r and cur is None:
            cur = i
        elif not r and cur is not None:
            if i - cur > best[1] - best[0] or best == (0, h):
                best = (cur, i)
            cur = None
    a, b = best
    m = max(1, int((b - a) * 0.08))
    a, b = max(0, a - m), min(h, b + m)
    if b - a < h * 0.3:
        return win, bw
    return win[a:b], bw[a:b]

def split_digits(win, n):
    """等分割を基本に、列投影の谷で境界を微調整"""
    # 背景が明るい場合は反転して「白字」に統一
    if win.mean() > 127:
        win = 255 - win
    _, bw = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    win, bw = crop_main_band(win, bw)
    h, w = win.shape
    step = w / n
    # 列投影のピーク（＝各数字の中心）を探し、そこにセル中心を合わせる。
    # 検出枠の左右余白で等分割がずれるのを防ぐ。
    proj = (bw > 0).astype(np.float32).sum(0)
    k = max(3, int(step * 0.6)) | 1
    sm = cv2.GaussianBlur(proj[None, :], (k, 1), 0)[0]
    centers = _pick_peaks(sm, n, min_dist=step * 0.6)
    if centers is None:
        centers = [(i + 0.5) * step for i in range(n)]
    # セル幅は中心間隔の中央値（右端の赤桁が広い等の機種差を吸収）
    cw = np.median(np.diff(centers)) if n > 1 else step
    cells = []
    for c in centers:
        a, b = int(round(c - cw / 2)), int(round(c + cw / 2))
        pad_l, pad_r = max(0, -a), max(0, b - w)
        cell = win[:, max(0, a):min(w, b)]
        if pad_l or pad_r:
            cell = cv2.copyMakeBorder(cell, 0, 0, pad_l, pad_r, cv2.BORDER_REPLICATE)
        cells.append(cv2.resize(cell, (W, H), interpolation=cv2.INTER_AREA))
    return cells, win

def _pick_peaks(sig, n, min_dist):
    """貪欲に高いピークから n 個、min_dist 以上離して選ぶ。左から順に返す。"""
    idx = np.argsort(-sig)
    chosen = []
    for i in idx:
        if sig[i] <= 0:
            break
        if all(abs(i - c) >= min_dist for c in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return None
    return sorted(chosen)

# ---------- 5. 分類 ----------
def classify(sess, cells, tta=True):
    """少し位置をずらした複数パッチで推論して平均（TTA）。数msで済む。"""
    x = np.stack(cells).astype(np.float32)[:, None] / 255.0
    views = [x]
    if tta:
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            views.append(np.roll(x, (dy, dx), axis=(2, 3)))
    probs = []
    for v in views:
        logits = sess.run(None, {"x": v})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
    prob = np.mean(probs, 0)
    return prob.argmax(1), prob.max(1)

def to_reading(labels, probs, red_digits=0):
    """
    回転中 (10+d) は小さい方 d を採用する（検針値は切り捨て）。
    上位桁が回転中なら、それより右の桁は 9→0 の途中で 9 側にいるはずなので、
    整合性のため右側の桁は 9 に揃える（例: [1, 回転2→3, 回転9→0] → 129）。
    """
    digits = []
    rolling = []
    for l in labels:
        l = int(l)
        rolling.append(l >= 10)
        digits.append(l - 10 if l >= 10 else l)
    first_roll = next((i for i, r in enumerate(rolling) if r), None)
    if first_roll is not None:
        for j in range(first_roll + 1, len(digits)):
            digits[j] = 9
    s = "".join(str(d) for d in digits)
    if red_digits:
        s = s[:-red_digits] + "." + s[-red_digits:]
    return s, float(np.min(probs)), rolling

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--digits", type=int, default=8, help="数字窓の桁数")
    ap.add_argument("--red", type=int, default=3, help="小数桁（赤字）の数")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ocr = load_detector()
    sess = ort.InferenceSession("digit_cnn.onnx", providers=["CPUExecutionProvider"])

    t0 = time.perf_counter()
    boxes = detect_boxes(ocr, img)
    t1 = time.perf_counter()
    p, score = pick_counter_window(gray, boxes, a.digits)
    if p is None:
        print("数字窓が見つかりません"); return
    win = rectify(gray, p)
    cells, win_n = split_digits(win, a.digits)
    labels, probs = classify(sess, cells)
    t2 = time.perf_counter()

    reading, conf, rolling = to_reading(labels, probs, a.red)
    print(f"reading: {reading}  min_conf: {conf:.2f}  window_score: {score:.2f}  rolling: {rolling}")
    print(f"detect {1000*(t1-t0):.0f} ms / rest {1000*(t2-t1):.0f} ms")
    print("per-digit:", [(int(l), round(float(c), 2)) for l, c in zip(labels, probs)])

    if a.debug:
        vis = img.copy()
        cv2.polylines(vis, [p.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite("debug_box.jpg", vis)
        cv2.imwrite("debug_window.png", win_n)
        cv2.imwrite("debug_cells.png", np.hstack(cells))
        print("debug_*.{jpg,png} を出力しました")

if __name__ == "__main__":
    main()
