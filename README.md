# Flounder Area Measure Program
　Flounder Area Measurement Program using YOLOv5
## 概要
YOLOv5を用いたヒラメの面積自動測定プログラムです。  
ヒラメ個体と1cm四方の基準四角を同じ画像内に撮影し、自動的に面積(cm²)を算出します。

## フォルダ構成
```
flounder_area_measure/
├── menseki.py           # メインプログラム
├── model.pt             # 学習済みモデル
├── requirements.txt     # 必要ライブラリ
├── data/                # 入力画像フォルダ
│   └── images/         # ヒラメ画像と基準四角画像
└── runs/                # 結果出力フォルダ（実行時に自動生成）
```

## セットアップ
```bash
git clone https://github.com/Sugai-Atsushi026/flounder_area_measure.git
cd flounder_area_measure
pip install -r requirements.txt
```

## 実行方法
```bash
python menseki.py --weights model.pt --source data/images/
```

## 出力
`runs/detect/exp/` フォルダに  
- 結果画像  
- 面積データCSV  

が保存されます。

## 備考
YOLOv5 本体は以下から別途インストールしてください。  
https://github.com/ultralytics/yolov5

