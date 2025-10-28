# MNIST classification

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*SfRJNb5dOOPZYEFY5jDRqA.png" alt="mnist" title="mnist">
</div>

今回はPython深層学習系ライブラリPytorchを使用してMNISTの分類とモデルの評価を行います。

### 深層学習とは
深層学習とは人間の脳の神経回路を模倣した多層構造の「ニューラルネットワーク」を用いて、大量のデータからルールやパターンを学習する機械学習の一種。
### MNISTとは
MNISTとは「Modified National Institute of Standards and Technology database」の略で、手書き数字の画像データセット。機械学習、特にニューラルネットワークを用いた画像認識の入門やベンチマークとして広く利用されている。
<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png" alt="mnist" title="mnist">
</div>

## セットアップ
### 各Docker環境へSSH接続
200xxは⾃分のSSHポート番号に変更する
```
ssh root@swelab1.mc.yc.tcu.ac.jp -p 200xx
```
### リポジトリのクローン
```
cd /work
git clone https://github.com/aiueo5938/mnist-classification.git
cd ./mnist-classification
```
### 仮想環境の作成と有効化
```
pyenv virtualenv 3.13.7 torch
pyenv local torch
```
### [Pytorch](https://pytorch.org/get-started/locally/)のインストール
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
### プログラムの実行
```
python ./main.py
```