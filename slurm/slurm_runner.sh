#!/bin/sh
python3 -m venv l-ki
source l-ki/bin/activate

pip3 install -r ../requirements.txt
cd ..
chmod -R 755 ./unity-env/*.x86_64

python3 main.py -env unity -env_n 3DBall1 -tag slurm-test -e 10