# MNIST classification

## セットアップ

### リポジトリのクローン
```
cd /work
git clone https://github.com/aiueo5938/mnist-classification.git
cd ./mnist-classification
```
### 仮想環境の作成と有効化
```
pyenv virtualenv 3.13.4 torch
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