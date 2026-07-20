curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
conda create -n demucs python=3.10 ffmpeg -y
conda activate demucs

pip install -e .
pip install "numpy<2"

python -m demucs.separate -n htdemucs -d cpu --two-stems vocals --mp3 test.mp3
