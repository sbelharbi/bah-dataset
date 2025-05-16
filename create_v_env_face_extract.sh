#!/usr/bin/env bash
local="~"
env="bah-main-face-extract"


cdir=$(pwd)



echo "Create virtual env $env"
if test -d $local/venvs/$env; then
  echo "Deleting" $local/venvs/$env
  rm -r $local/venvs/$env
fi

python3.9.19 -m venv $local/venvs/$env
source $local/venvs/$env/bin/activate
pip install --upgrade pip


python --version

pip install texttable more-itertools

echo "Installing..."

pip install pyparsing attrs certifi click requests jinja2 markupsafe pyyaml typing-extensions
pip install munch
pip install pynvml

pip install numpy==1.23

pip install https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl#sha256=5422d19042e217c2aa94030b16b3fe4da5be9ba8eea46e7e59d40a110955962d
pip install https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl#sha256=00e7e19c74e155d271e6ac63d8b4e3f93f5d3cc171428c4fdddbb1226d72eca1

pip install seaborn
pip install tqdm matplotlib scipy pandas  scikit-learn ipython opencv-python

pip install retinaface-pytorch==0.0.8

pip install ffmpeg-python
pip install face-alignment
pip install hume==0.7.5

deactivate

echo "Done creating and installing virt.env: $env."
