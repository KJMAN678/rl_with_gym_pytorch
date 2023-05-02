## 開発環境

- M1 Mac Venture 13.3.1(a) / SOROBAN nvd5-1ub Ubuntu18.04
- Python 3.8.16

```sh
# SOROBAN 環境作成スクリプト
bash create_python38_env_for_soroban.sh

# pyenv の環境変数設定
nano ~/.bashrc

# 以下を .bashrc に書き込む
PYENV_ROOT="${HOME}/.pyenv"
PATH="${PYENV_ROOT}/bin:${PATH}"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

# 再起動
exec $SHELL

# pyenv で Python 3.8.16 をインストール
pyenv install 3.8.16

# Python 3.8.16 に変更
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
