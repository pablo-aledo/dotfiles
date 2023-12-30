cd
git clone https://github.com/trholding/llama2.c.git
cd llama2.c
make run_cc_fast
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
./run stories15M.bin
