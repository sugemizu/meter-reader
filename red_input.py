"""
=====================================================================
red_input.py — 赤い小数桁のセルを CNN の入力にする前処理（学習と推論で共通）
=====================================================================

■ なぜ別ファイルか
  学習側（synth_red_digits.py）と推論側（read_meter.py）が「まったく同じ関数」を使う必要がある。
  read_meter.py に置くと、合成データの生成が onnxruntime や検出器の import まで引きずるため分けた。
  numpy だけに依存するので、端末へ移植するときもこの数行を書き直すだけで済む。

■ 何をするか
  赤インクの強さ score = R - max(G, B) を使う。灰・黒・白（箱の枠、目盛り帯、▶、黒い文字盤）は
  R≒G≒B なので score が 0 になって消え、赤い字だけが「黒地に明るい字」で残る（黒桁モデルと同じ極性）。
  以前の「反転グレー」は、白い箱の中の暗い灰・箱の外の明るい灰・目盛り帯が字と同程度の
  コントラストで入り、字そのものは薄かった（README の「赤い小数桁の専用モデル」参照）。
"""
import numpy as np

RED_MIN_RANGE = 16.0   # 赤みの幅がこれ未満のセルは「伸ばしすぎない」（雑音を 255 まで引き伸ばさない）
RED_EMPTY = 8.0        # 赤みの幅がこれ未満なら「赤が無い」とみなし、推論側は従来の黒モデル + 反転グレーに戻す
                       # （ベンチ 228 件の赤セルは最小でも 13 なので、この値はベンチでは確かめられていない）


def red_score_range(bgr):
    """R - max(G, B) の 99% 点 - 5% 点（赤インクがどれだけ際立つか）と、その score 配列を返す"""
    b, g, r = [bgr[..., i].astype(np.float32) for i in range(3)]
    s = np.clip(r - np.maximum(g, b), 0, None)
    return float(np.percentile(s, 99) - np.percentile(s, 5)), s


def red_cell_input(bgr):
    """
    28x40 の BGR セル → 赤インクが明るい 28x40 の uint8（黒地に明るい字）。
    入力は必ず BGR（OpenCV の順）。RGB を渡すと青インクを学習してしまう。
    score を 5%/99% 点で 0〜255 に伸ばす。幅が RED_MIN_RANGE 未満なら RED_MIN_RANGE で割って、
    赤みの無いセルの雑音を引き伸ばさないようにする。
    二値化しないのは、細い線が切れるのを避けるため（debug_redmask.png の二値マスクは線が切れる）
    """
    rng, s = red_score_range(bgr)
    lo = float(np.percentile(s, 5))
    return np.clip((s - lo) / max(rng, RED_MIN_RANGE) * 255.0, 0, 255).astype(np.uint8)
