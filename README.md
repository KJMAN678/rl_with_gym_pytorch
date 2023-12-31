## 開発環境

- M1 Mac Venture 13.3.1(a) / SOROBAN nvd5-1ub Ubuntu18.04
- Python 3.10.11

```sh
- SOROBAN用の初回のみ、環境構築
# SOROBAN 環境作成スクリプトを実行
bash create_python310_env_for_soroban.sh

# pyenv の環境変数設定
nano ~/.bashrc

# 下記を追記して保存
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

# 追記した環境変数を認識させる
source ~/.bashrc

# pyenv で Python 3.8.16 をインストール
pyenv install 3.10.11

# Python 3.8.16 に変更
pyenv local 3.10.11
```

```sh
- Mac / SOROBAN 共通 仮想環境の構築

# 仮想環境の作成
python3 -m venv .venv

# 仮想環境のアクティベート
source .venv/bin/activate

# pip のアップグレード
python3 -m pip install --upgrade pip

# ライブラリのインストール
pip install -r requirements.txt
```

## 参考書

- [Book:Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym](https://www.amazon.co.jp/dp/B091K32T2B)
- [GitHub](https://github.com/Apress/deep-reinforcement-learning-python)

## 参考サイト

- [Docker と Docker Compose で Python の実行環境を作成する](https://zuma-lab.com/posts/docker-python-settings)
- [docker の python コンテナをビルドしなおさず pip install でライブラリを更新できるようにする方法](https://asukiaaa.blogspot.com/2020/07/docker-python-pip-install-without-rebuilding.html)
- [gym 0.21 がインストールできない](https://github.com/openai/gym/issues/3176)
  - setuptools のバージョンが原因
- [【Linux】Ubuntu 22.10(Kinetic Kudu)に pyenv をインストールし環境構築する
  ](https://namileriblog.com/linux/ubuntu_pyenv/#i-3)
- [how to solve module 'gym.wrappers' has no attribute 'Monitor'？](https://stackoverflow.com/questions/71411045/how-to-solve-module-gym-wrappers-has-no-attribute-monitor)
- [OpenCV を使用して Python で動画（ビデオ）を再生する](https://laboratory.kazuuu.net/play-video-in-python-using-opencv/)
- [ALE Namespace does not exist.](https://github.com/openai/gym/issues/3201#issuecomment-1493032556)
- [Error in importing environment OpenAI Gym](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)
- [zsh でのコマンド入力時の注意](https://ttt24224222.hatenadiary.jp/entry/2018/06/30/181130)
- [Gym で使用できる環境の確認](https://github.com/openai/gym/issues/3201#issuecomment-1487606973)
- Gymnasium で Atari の ROM を使う
  - [AutoROM --accept-license](https://github.com/openai/gym/issues/3170#issuecomment-1377978144)
  - [pip install gymnasium[atari, accept-rom-license]](https://github.com/openai/gym/issues/3201#issuecomment-1493032556)
- [ubuntu で pyenv 使用を試みるも python バージョンが切り替わらない](https://qiita.com/seigot/items/dc63ed75e6a46f1accab)
