#!/usr/bin/env bash
local="~"
env="bah-main"


cdir=$(pwd)




echo "Create virtual env $env"
if test -d $local/venvs/$env; then
  echo "Deleting" $local/venvs/$env
  rm -r $local/venvs/$env
fi

python3.10.14 -m venv $local/venvs/$env
source $local/venvs/$env/bin/activate
pip install --upgrade pip


python --version


pip install texttable more-itertools

echo "Installing..."

pip install pyparsing attrs certifi click requests jinja2 markupsafe pyyaml typing-extensions
pip install munch
pip install pynvml


pip install numpy==1.26.0
# these versions are required by facenet-pytorch 2.6.0
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install seaborn
pip install tqdm matplotlib scipy pandas  scikit-learn ipython openpyxl


pip install facenet-pytorch==2.6.0

pip install ffmpeg-python
pip install face-alignment

pip install opencv-python tqdm opensmile resampy vosk
pip install --upgrade nltk

pip install deepmultilingualpunctuation
pip install ffmpeg-python
python -c 'import nltk; nltk.download("punkt")'


pip install mlcroissant
pip install gitpython


cd $cdir

deactivate

echo "Done creating and installing virt.env: $env."
