cd
git clone https://github.com/tairov/llama2.py.git
cd llama2.py
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
python3 llama2.py stories15M.bin 0.8 256 "Dream comes true this day"
