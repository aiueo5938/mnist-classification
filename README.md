# MNIST classification

<div align="center">
    <img src="https://user-images.githubusercontent.com/68801296/88917938-4008f180-d286-11ea-8667-50027700e3ea.png" alt="mnist" title="mnist">
</div>
<div align="right">
    出典：Image-Classification-with-MNIST-Dataset-using-keras - GitHub
</div>

<br>
今回はPython深層学習系ライブラリPytorchを使用してMNISTの分類とモデルの評価を行います。

### 深層学習とは
深層学習とは人間の脳の神経回路を模倣した多層構造の「ニューラルネットワーク」を用いて、大量のデータからルールやパターンを学習する機械学習の一種。
### MNISTとは
MNISTとは「Modified National Institute of Standards and Technology database」の略で、手書き数字の画像データセット。機械学習、特にニューラルネットワークを用いた画像認識の入門やベンチマークとして広く利用されている。

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png" alt="mnist" title="mnist">
</div>
<div align="right">
    出典：IMNIST database - Wikipedia
</div>

### モデルの評価方法
機械学習における代表的な評価指標としてAccuracy, Recall Precision, F1がある。
今回はAccuracyの算出を実際にプログラムで体験してもらう。
| 略称  | 英語名             | 日本語訳   | 意味                       |
|------|------------------|----------|-------------------------------|
| TP   | True Positive    | 真陽性    | 本当は陽性で、予測も陽性と判定した     |
| TN   | True Negative    | 真陰性    | 本当は陰性で、予測も陰性と判定した     |
| FP   | False Positive   | 偽陽性    | 本当は陰性だが、予測は陽性と判定した   |
| FN   | False Negative   | 偽陰性    | 本当は陽性だが、予測は陰性と判定した   |

| 指標            | 日本語訳     | 定義・数式                            | 主な意味・使いどころ                            |
|----------------|------------|-------------------------------------|-------------------------------------------|
| **Accuracy**   | **正解率**   | ![accuracy](images/accuracy.svg)　  | **全ての予測のうち正しく予測できたものの割合**       |
| Recall    　　  | 再現率       | ![recall](images/recall.svg)       | 実際に正であったもののうち、どれだけ「正」と予測できたか |
| Precision 　　  | 適合率       | ![precision](images/precision.svg) | 「正」と予測したうち、実際に正であった割合           |
| F1-score  　　  | F1スコア     | ![f1](images/f1.svg)               | Precision・Recallの調和平均。バランス重視に使う    |

<!-- | 指標       | 日本語訳     | 定義・数式                                            | 主な意味・使いどころ                     |
|-----------|------------|----------------------------------------------------|------------------------------------------|
| ***Accuracy*** | ***正解率*** | \( \frac{TP + TN}{TP + TN + FP + FN} \)　| ***全ての予測のうち予測が正しかった割合***           |
| Recall    　　| 再現率      | \( \frac{TP}{TP + FN} \)                           | 実際に正であったもののうち、どれだけ「正」と予測できたか |
| Precision 　　| 適合率      | \( \frac{TP}{TP + FP} \)                           | 「正」と予測したうち、実際に正であった割合|
| F1-score  　　| F1スコア    | \( \frac{2 \times Precision \times Recall}{Precision + Recall} \) | Precision・Recallの調和平均。バランス重視に使う       | -->

<!-- \( \frac{予測が正しかった数}{予測したデータ数} \) -->

## セットアップ
### Docker環境へSSH接続
200xxは⾃分のSSHポート番号に変更する。
「2025事例研⽣向け 増⽥研究室 サーバー環境の使い⽅」を参照してください。
```
ssh root@swelab1.mc.yc.tcu.ac.jp -p 200xx
```
<!-- ### Jupyterの起動

89xxは⾃分のSSHポート番号に変更する
```
nohup jupyter lab --port=89xx --ip=0.0.0.0 --allow-root --NotebookApp.token='' >/dev/null 2>&1 &
``` -->
### リポジトリのクローン
```
cd /work
git clone https://github.com/aiueo5938/mnist-classification.git
cd ./mnist-classification
```
<!-- ### 仮想環境の作成と有効化
```
pyenv virtualenv 3.13.7 torch
pyenv local torch
``` -->
### pipのアップグレード
```
python -m pip install --upgrade pip
```
### [Pytorch](https://pytorch.org/get-started/locally/)のインストール
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
### プログラムの実行
```
python ./main.py
```