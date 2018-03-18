<!-- -*- encoding: utf-8 -*- -->
# Wasserstein AutoEncoder

https://github.com/tolstikhin/wae のfork

## 修正点

* Python3で動作するよう修正
* mnistのデータセットを取得するツールの追加 (download.py)
* 64x64カラー画像データを扱うexperimentの追加(--exp dir64)
  * dir64以下のディレクトリに置かれた画像を訓練対象とする
* ランダムに画像を生成するスクリプトの追加 (gen.py)
* 2つの画像の潜在空間の中間点を含めた画像の生成スクリプト追加 (analogy.py)
