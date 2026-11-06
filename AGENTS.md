# AGENTS.md — meter-reader

AI エージェント（Claude Code / Codex / その他）がこのリポジトリで作業するときの共通ガイド。
Claude Code 固有の指揮ルールは `CLAUDE.md` を参照。

## プロジェクト概要

水道メーターの数字窓を写真から読み取る PoC。

- 検出: PP-OCR DB（rapidocr-onnxruntime 同梱の ONNX）で文字領域を検出し、`pick_counter_window` で数字窓を選ぶ
- 認識: ドラム帯を模した合成データで学習した 3 層 CNN（`digit_cnn.onnx`）を ONNX Runtime で実行
- 回転中の桁は「小さい方の値を採用し、右側の桁は 9 に揃える」切り捨て読み

詳しい手法・実写での調整履歴は `README.md` にある。README は「何を試して何が効いたか」の記録でもあるので、
手法を変えたら README も更新する。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `read_meter.py` | 推論の本体。検出 → 窓選択 → 正面化 → 桁分割 → CNN 分類 → 読み値生成 |
| `synth_digits.py` | 合成データ生成（`data/synth/`）。フォント検索、手描き数字（平らな 3、直線の 6/9、開いた 4）、augmentation |
| `train_digit_cnn.py` | CNN 学習と ONNX 書き出し |
| `digit_cnn.onnx` | 学習済み認識モデル（コミット済み。再学習しない限り触らない） |
| `requirements.txt` | torch / onnx / onnxruntime / opencv-python-headless / pillow / rapidocr-onnxruntime |

`data/`, `in/`, `debug_*`, `*.pt` は `.gitignore` 済み。生成物やテスト画像をコミットしない。

## コマンド

```bash
pip install -r requirements.txt
python synth_digits.py                              # 合成データ生成
python train_digit_cnn.py                           # 学習 → digit_cnn.onnx
python read_meter.py photo.jpg --digits 5 --red 1 --debug   # 推論
```

- `--digits` 数字窓の桁数、`--red` 右端の赤い小数桁の数、`--invert` 白地に黒文字の機種、`--debug` で中間画像を出力
- 自動テストは無い。動作確認は実写画像で `read_meter.py` を回し、README の「実写での結果」表と比較する
- 学習は CPU でも回るが時間がかかる。頼まれていない限り再学習・モデル差し替えはしない

## コーディング規約

- Python。既存ファイルは行末・ブロック頭に日本語コメントで「なぜそうしたか」を書いている。新規コードも同じ流儀で
- 新しいライブラリを足す前に、既存依存（cv2 / numpy / PIL / onnxruntime）で済まないか確認する
- 画像処理のしきい値やマジックナンバーは、実写で決めた経緯をコメントに残す（README の調整履歴と対応させる）
- 「試して戻した」変更は README の該当節に 1 行残す（同じ失敗を繰り返さないため）
- コミットメッセージは日本語、1 行目で「何を直したか」を書く（既存履歴に合わせる）

## モデルの役割分担

複数モデルで作業する場合の基本方針。Claude Code での具体的な呼び方は `CLAUDE.md` にある。

| 役割 | モデル | 担当 |
|---|---|---|
| 指揮 | Fable | 要求の整理、タスク分解、作業指示、結果の検証、ユーザーへの報告。**自分ではコードを書かない** |
| 重い推論・調査 | Opus | アルゴリズム設計、誤認識の原因分析、複数ファイルにまたがる調査、設計レビュー、難しいバグの切り分け |
| 実装・実行 | Sonnet | コードの修正・追加、スクリプト実行、学習・推論の実行、README 更新、コミット作成 |
| 軽作業 | Haiku | ファイル検索、grep、ログや出力の要約、整形、簡単な確認作業 |

原則:

1. 指揮役は実装に手を出さず、「何を・どこを・どう確認するか」を書いて実行役に渡す
2. 判断に迷う調査は Opus、手順が明確な作業は Sonnet、探すだけ・まとめるだけは Haiku
3. 実行役は作業結果（変更したファイル、実行したコマンド、出力、未解決点）をそのまま報告し、指揮役が検証する
4. 1 つのタスクに 1 エージェント。並列化できる独立作業は同時に投げる
