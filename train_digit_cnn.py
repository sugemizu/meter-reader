"""
=====================================================================
train_digit_cnn.py — 合成画像で「1桁の数字を読む CNN」を学習する
=====================================================================

■ このファイルの役割
  synth_digits.py が作った画像（data/synth/）を読み込み、
  小さなニューラルネット（CNN）に「この画像はどの数字か」を覚えさせる。
  学習が終わったら、アプリから使える形式 (ONNX) で digit_cnn.onnx に保存する。

■ 入出力
  入力 : 1枚のグレースケール画像 28×40（0〜1 の小数）
  出力 : 20 個のスコア（0〜9 = 静止した数字、10+d = d→d+1 の回転中）
         一番スコアが高いものが答え

■ 使い方
  python train_digit_cnn.py            # 12 エポック学習
  EPOCHS=4 python train_digit_cnn.py   # 短く試すとき

  赤い小数桁専用モデルは、環境変数で入出力を切り替えて同じスクリプトで学習する
  （引数を足すより差分が小さく、既定値では今までと完全に同じ動作になる）:
    DATA=data/synth_red OUT=digit_cnn_red.onnx FLIP=0 python train_digit_cnn.py
    DATA : 学習画像のフォルダ（既定 data/synth）
    OUT  : 書き出す ONNX のファイル名（既定 digit_cnn.onnx。.pt も同じ名前で拡張子だけ替わる）
    FLIP : 1 なら学習中に 50% で明暗反転（既定）。赤みチャネルは常に黒地に明るい字なので 0 にする
    SEED : 指定したときだけ乱数を固定する

■ 用語
  エポック   : 全データを 1 周すること
  損失(loss) : 「どれくらい間違えているか」の数値。小さいほど良い
  val_acc    : 学習に使っていないデータでの正解率
"""

import glob, random, os
import numpy as np
import torch                 # PyTorch（ニューラルネットのライブラリ）
import torch.nn as nn        # ネットワークの部品（層）が入っている
from PIL import Image

# ================== 設定値 ==================
DATA = os.environ.get("DATA", "data/synth")          # 合成画像のフォルダ（赤桁用は data/synth_red）
OUT = os.environ.get("OUT", "digit_cnn.onnx")        # 書き出す ONNX（赤桁用は digit_cnn_red.onnx。黒桁モデルを上書きしないこと）
FLIP = os.environ.get("FLIP", "1").lower() not in ("0", "false", "no")   # 明暗反転の augmentation（赤みチャネルは極性が一定なので赤桁用は 0）
# 赤桁用データで黒桁モデル（コミット済みの digit_cnn.onnx）を上書きする事故を防ぐ
if DATA != "data/synth" and OUT == "digit_cnn.onnx":
    raise SystemExit("DATA が既定 (data/synth) 以外なのに OUT が既定の digit_cnn.onnx のままです。黒桁モデルを上書きしないよう OUT を指定してください")
W, H = 28, 40                                        # 画像サイズ（synth_digits.py と同じ値）
EPOCHS = int(os.environ.get("EPOCHS", 12))           # 環境変数 EPOCHS があればそれを使う。無ければ 12


# ================== ネットワークの定義 ==================
class DigitNet(nn.Module):
    """
    3 層の畳み込み（Conv）＋ 2 層の全結合（Linear）という、ごく小さい CNN。
    パラメータ数は数万〜十数万程度で、スマホでも 1 桁あたり 1ms 未満で動く。
    """
    def __init__(self, n=20):   # n = 出力クラス数（20）
        super().__init__()

        def blk(i, o):
            """
            「畳み込み → 正規化 → 活性化 → 縮小」の 1 セット。
            i: 入力チャンネル数, o: 出力チャンネル数
              Conv2d    : 3×3 のフィルタで特徴（線・角など）を抽出
              BatchNorm : 値のばらつきをそろえて学習を安定させる
              ReLU      : マイナスを 0 にする活性化関数
              MaxPool   : 縦横を半分に縮小（2×2 の中の最大値だけ残す）
            """
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        # 特徴抽出部: 1ch → 16ch → 32ch → 64ch。画像サイズは 40×28 → 20×14 → 10×7 → 5×3 と縮む
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))

        # 分類部:
        #   Flatten : 64ch × 5 × 3 の立体を 1 列（960 個）に並べ直す
        #   Dropout : 学習中にランダムに 30% を無効化して「暗記」を防ぐ
        #   Linear  : 960 → 128 → n（クラス数）
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 5 * 3, 128), nn.ReLU(), nn.Linear(128, n))

    def forward(self, x):
        """画像 x を入れると 20 個のスコア（logits）が出てくる"""
        return self.c(self.f(x))


# ================== データ読み込み ==================
def load():
    """
    DATA（既定 data/synth）/<ラベル>/*.png を全部読み込んで、
    (学習用 90%, 検証用 10%) に分けて返す。
    """
    xs, ys = [], []    # xs: 画像の配列, ys: ラベル（正解）
    for label in range(20):
        for p in glob.glob(f"{DATA}/{label}/*.png"):
            # グレースケールで読んで 0〜255 → 0〜1 の小数に変換
            xs.append(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0)
            ys.append(label)

    # PyTorch 用のテンソル（多次元配列）に変換。
    # [:, None] で (枚数, 高さ, 幅) → (枚数, 1, 高さ, 幅) にして「チャンネル数 1」の次元を足す
    x = torch.tensor(np.stack(xs))[:, None]
    y = torch.tensor(ys)

    # 順番をランダムにシャッフル（ラベル順に並んだままだと偏るため）
    idx = torch.randperm(len(y))
    x, y = x[idx], y[idx]

    # 先頭 90% を学習用、残り 10% を検証用に
    n = int(len(y) * 0.9)
    return (x[:n], y[:n]), (x[n:], y[n:])


# ================== 学習 ==================
def main():
    if "SEED" in os.environ:             # 指定時のみ固定（既定は今までどおり未固定）
        random.seed(int(os.environ["SEED"])); torch.manual_seed(int(os.environ["SEED"]))
    (xtr, ytr), (xva, yva) = load()      # tr = train（学習用）, va = validation（検証用）
    net = DigitNet()                     # ネットワークを作る（最初はでたらめな重み）

    # 最適化アルゴリズム: Adam。学習率 0.001
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    # 学習率を徐々に下げるスケジューラ（終盤は細かく調整させる）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    # 損失関数: 分類問題の定番「交差エントロピー」
    lossf = nn.CrossEntropyLoss()
    bs = 128                             # バッチサイズ（一度に何枚まとめて学習するか）

    for ep in range(EPOCHS):
        net.train()                      # 学習モード（Dropout などが有効になる）
        perm = torch.randperm(len(ytr))  # 毎エポック順番をシャッフル

        # 128 枚ずつ取り出して学習
        for i in range(0, len(ytr), bs):
            b = perm[i:i + bs]           # このバッチに使う画像の番号
            xb = xtr[b]
            if FLIP and random.random() < 0.5:    # 50% の確率で明暗を反転（白地黒字にも強くする）
                xb = 1 - xb

            opt.zero_grad()                       # 前回の勾配をリセット
            loss = lossf(net(xb), ytr[b])         # 予測と正解を比べて損失を計算
            loss.backward()                       # 損失が減る方向（勾配）を計算
            opt.step()                            # 重みを少し更新

        sched.step()                     # 学習率を 1 段階下げる

        # ---- 検証: 学習に使っていない 10% で正解率を測る ----
        net.eval()                       # 評価モード（Dropout を切る）
        with torch.no_grad():            # 勾配計算をしない（速い・メモリ節約）
            # argmax(1) = 20 個のスコアのうち最大の番号 → 予測ラベル
            acc = (net(xva).argmax(1) == yva).float().mean().item()
        print(f"epoch {ep+1} loss {loss.item():.3f} val_acc {acc:.4f}")

    # ---- 保存 ----
    torch.save(net.state_dict(), os.path.splitext(OUT)[0] + ".pt")     # PyTorch 形式（再学習用）

    # ONNX 形式で書き出す。ONNX は PyTorch が無い環境（スマホ等）でも動かせる共通形式。
    #   torch.zeros(1,1,H,W) : 入力の形を教えるためのダミー画像
    #   dynamic_axes         : 1 枚でも 8 枚でもまとめて推論できるように「枚数」を可変にする
    torch.onnx.export(net, torch.zeros(1, 1, H, W), OUT,
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
    print("saved", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
