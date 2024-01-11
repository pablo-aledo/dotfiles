cd
mkdir minillm
cd minillm

conda create -n minillm
conda activate minillm
conda install git pip virtualenv
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit

wget https://huggingface.co/kuleshov/llama-30b-4bit/resolve/main/llama-30b-4bit.pt
wget https://huggingface.co/kuleshov/llama-65b-4bit/resolve/main/llama-65b-4bit.pt
