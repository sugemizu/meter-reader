# 水道メーター読み取り PoC（方式A: 学習済み検出器 + 合成データCNN）

## 手順
```
pip install -r requirements.txt
python synth_digits.py        # 合成データ生成 (data/synth/)
python train_digit_cnn.py     # 学習 → digit_cnn.onnx (約20KB)
python read_meter.py photo.jpg --digits 8 --red 3 --debug
```
`--digits` は数字窓の桁数、`--red` は右端の小数桁（赤字）の数。
`--debug` で検出枠 (debug_box.jpg)、正面化した窓 (debug_window.png)、分割した各桁 (debug_cells.png) を出力。

## 構成
- 検出: PP-OCR DB (MobileNet) 学習済み ONNX（rapidocr-onnxruntime 同梱） → `pick_counter_window` でアスペクト比・暗さ・サイズでスコアリング
- 認識: ドラム帯をシミュレートした合成データ (0-9 静止 / 10+d 回転中) で学習した3層CNN、ONNX Runtime で実行
- 回転中は小さい方の値 d を採用し、その右側の桁は 9 に揃える（切り捨て読み）
- 桁分割は列投影のピークに各セル中心を合わせる（検出枠の余白によるズレ対策）

## 端末移植
- 検出器: rapidocr 同梱の det ONNX モデル (約4.7MB) をそのまま ONNX Runtime Mobile へ
- 認識器: digit_cnn.onnx をそのまま。iOS は Core ML 変換 (coremltools) でも可

## 実写1枚での結果
手元の実写 (黒4桁+赤1桁, 0723.1) で `--digits 5 --red 1` → 0723.1、全桁の信頼度 0.84 以上。
検出 ~350ms (PC/CPU, 640px)、それ以降 ~6ms。

## 合成データで効いた調整（実写を見て直した点）
- ドラム帯から窓を切り出す生成方式（上下に隣の数字がのぞく状態を「静止」として学習）
- 数字の高さをピッチの 62〜95% に（実機は密）
- 太字: stroke_width で肉付け
- 上が平らな「3」を描画で追加（フォントの丸い3だけだと 5 と誤認）
- 小数点・隣セルのドットが角に入る augmentation
