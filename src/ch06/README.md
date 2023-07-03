```sh
python3 src/ch06/torch_dqn.py

AutoROM --accept-license # (初回のみ実行)ライセンスの受け入れ処理
python3 src/ch06/torch_dqn_atari_game.py

python3 src/ch06/torch_dqn_prioritized_replay.py

python3 src/ch06/torch_double_dqn.py
```

# tkinter のランタイムエラー対策

- デフォルトの tkinter の tcl-tk のバージョンが 8.5 だが、Mac には tcl-tk 8.6 が入っているため、下記のエラーが起きる

```sh
RuntimeError: tk.h version (8.5) doesn't match libtk.a version (8.6)
```

- そこで、tkinter 用の環境変数を設定する
- tcl-tk がインストールされている場所を探し、include フォルダと lib フォルダのパスを調べる
- 下記の TCL_INCLUDE, TCL_LIB のパスを適宜変更する

```sh
TCL_INCLUDE="opt/homebrew/Cellar/tcl-tk/8.6.13_2/include"
TCL_LIB="opt/homebrew/Cellar/tcl-tk/8.6.13_2/lib"
PYTHON_CONFIGURE_OPTS="--with-tcltk-includes=$TCL_INCLUDE --with-tcltk-libs='$TCL_LIB -ltcl8.6 -ltk8.6'"
```

- 上記の設定をして pyenv install 3.X.X を実行し、Python のバージョンを変更
- 下記コマンドを実行し、下記画像のような GUI が開かれれば成功

```sh
python -m tkinter
```

<img src="../../images/ch06_tk_test_gui.png">

- [参考:Mac の pyenv 環境下で tkinter を使えるようにする方法](https://qiita.com/saki-engineering/items/92b7ec12ed07338929a3)
- [gymnasium.wrappers.RecordVideo](https://gymnasium.farama.org/main/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo)
  - [強化学習実装入門 (DQN 編)](https://unproductive.dev/rl-implementation-dqn/#%E5%AD%A6%E7%BF%92%E7%B5%90%E6%9E%9C)
