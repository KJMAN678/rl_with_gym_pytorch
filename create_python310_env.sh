#!/bin/bash

sudo apt update
sudo apt-get update -y
sudo apt-get upgrade -y

# for .venv python 3.10
sudo apt install python3.10-venv -y

# for gym 0.20.0
pip install setuptools==65.5.0

# pyenv install
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# pyenv dependency
sudo apt install libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev make gcc -y
sudo apt-get install liblzma-dev -v