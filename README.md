# Object Detection Project

このプロジェクトは、YOLOv8とMediaPipeを使用した物体検出・顔検出のWebアプリケーションです。

## プロジェクト構造

```
object_detection/
├── src/           # ソースコード
│   └── webapp/    # Streamlitベースのウェブアプリケーション
├── tests/         # テストコード
├── docs/          # ドキュメント
└── examples/      # 使用例
```

## セットアップ

1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

2. Webアプリの起動
```bash
cd src/webapp
streamlit run app.py
```

## 機能

- YOLOv8による物体検出
- MediaPipeによる顔検出
- リアルタイムカメラ入力対応
- 画像ファイルのアップロード対応 