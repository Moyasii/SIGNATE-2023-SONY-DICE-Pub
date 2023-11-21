# SIGNATE ソニーグループ合同 データ分析コンペティション NaN位 解法

このリポジトリには2023年にSIGNATEで開催された[ソニーグループ合同 データ分析コンペティション](https://signate.jp/courses/OJXBVN6v3M9RYvdZ)で以下のスコアを出したときの実験コードが登録されています（受賞対象外者なのでLBからは消えているかもしれません）。

* **Public LB**: 0.9930182
* **Private LB**: 0.9934195

解法は ①サイコロ部分の切り出しを行い、 ②切り出したサイコロの出目を予測するというシンプルな2段階構成です。

<img src=docs/solution.png width=800px>

全体のパイプラインは以下の通りです。

<img src=docs/pfd.png width=800px>

以降のセクションの通り環境構築をすることで、Docker上で実際にパイプラインを実行することができます。

## 確認環境

* Ubuntu 20.04
* NVIDIA GPU (RTX3090で確認)
* NVIDIA Driver
* Docker
* NVIDIA Container Toolkit

## 事前準備

実行に必要なライブラリをインストールする。

### NVIDIA Driverのインストール

<https://ubuntu.com/server/docs/nvidia-drivers-installation>を参照してNVIDIA Driverをインストールする。

```bash
sudo ubuntu-drivers list
sudo apt install nvidia-driver-525
```

### Dockerのインストール

<https://docs.docker.com/engine/install/ubuntu/>を参照してDockerをインストールする。

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install the Docker packages.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Manage Docker as a non-root user
sudo groupadd docker
sudo usermod -aG docker $USER
docker run hello-world
newgrp docker

# Check
docker run hello-world
```

### NVIDIA Container Toolkitのインストール

<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>を参照してNVIDIA Container Toolkitをインストールする。

```bash
# Configure the repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && sudo apt-get update

# Install the NVIDIA Container Toolkit packages:
sudo apt-get install -y nvidia-container-toolkit

# Check
docker run --gpus all --rm nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

## 実行手順

1. リポジトリのルートにX_train.npy, y_train.npy, X_test.npyをダウンロードする
2. Docker Imageを作成する
    ```bash
    docker image build -t sony_dice_image .
    ```
3. Dockerコンテナ上のJupyter Notebookサーバーを起動する
    ```bash
    docker run -it -p 8888:8888 --gpus all --rm sony_dice_image jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
    ```
4. ホストマシンのブラウザからJupyter Notebookサーバーに<http://127.0.0.1:8888/tree>にアクセスする
5. sony_dice.ipynbを起動し上から順番に実行する