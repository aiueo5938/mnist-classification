# MNIST classification

## セットアップ

### リポジトリのクローン
```
cd /work
git clone https://github.com/aiueo5938/mnist-classification.git
```
### 仮想環境の作成と有効化
```
pyenv virtualenv 3.13.4 torch
pyenv local torch
```
### Pytorchのインストール
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```