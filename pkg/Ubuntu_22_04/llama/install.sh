cd ~/tmp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
docker run -v /llama/models:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 7B
make
./main -m /llama/models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512
