```sh
# SOROBAN nvd5-1l22ul Ubuntu22.04 Python 3.10.6
# 環境作成スクリプト
bash create_python310_env.sh

# pyenv の環境変数設定
vi ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
# 再起動
exec $SHELL

# pyenv で Python 3.8.16 をインストール
pyenv install 3.8.16

# pyenv で Python 3.8.16 を設定
pyenv local 3.8.16

# 仮想環境の作成
python3 -m venv .venv

# 仮想環境のアクティベート
source .venv/bin/activate

# pip のアップグレード
python3 -m pip install --upgrade pip

# ライブラリのインストール
pip install -r requirements.txt
```


```sh
# Docker イメージ作成
docker-compose build

# Docker イメージの確認
docker images

# コンテナ起動
docker-compose up -d
```

- [Book:Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym](https://www.amazon.co.jp/dp/B091K32T2B)
- [GitHub](https://github.com/Apress/deep-reinforcement-learning-python)

## 参考サイト

- [Docker と Docker Compose で Python の実行環境を作成する](https://zuma-lab.com/posts/docker-python-settings)
- [docker の python コンテナをビルドしなおさず pip install でライブラリを更新できるようにする方法](https://asukiaaa.blogspot.com/2020/07/docker-python-pip-install-without-rebuilding.html)
- [gym 0.21 がインストールできない](https://github.com/openai/gym/issues/3176)
  - setuptools のバージョンが原因
- [【Linux】Ubuntu 22.10(Kinetic Kudu)にpyenvをインストールし環境構築する
](https://namileriblog.com/linux/ubuntu_pyenv/#i-3)