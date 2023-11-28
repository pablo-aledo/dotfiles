# get the model data
git lfs install
git clone https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B

# clone llama.cpp and setup python conversion stuff
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
ln -sfn ../OpenHermes-2.5-Mistral-7B ./models/openhermes-7b-v2.5

# convert to F16 GGUF
python3 convert.py ./models/openhermes-7b-v2.5 --outfile ./models/openhermes-7b-v2.5/ggml-model-f16.gguf --outtype f16

# quantize to Q8_0 and Q4_K
./quantize ./models/openhermes-7b-v2.5/ggml-model-f16.gguf ./models/openhermes-7b-v2.5/ggml-model-q8_0.gguf q8_0
./quantize ./models/openhermes-7b-v2.5/ggml-model-f16.gguf ./models/openhermes-7b-v2.5/ggml-model-q4_k.gguf q4_k

# build the benchmark tool
LLAMA_CUBLAS=1 make -j batched-bench

./batched-bench ./models/openhermes-7b-v2.5/ggml-model-f16.gguf  4096  0 99 0 2048 128,512 1,2,3,4 # bench the F16 model
./batched-bench ./models/openhermes-7b-v2.5/ggml-model-f16.gguf  4096  0 99 0 512  128,512 1,2,3,4 # bench the F16 model using small prompt size of 512
./batched-bench ./models/openhermes-7b-v2.5/ggml-model-q8_0.gguf 10240 0 99 0 2048 128,512 1,2,3,4 # bench the Q8_0 model
./batched-bench ./models/openhermes-7b-v2.5/ggml-model-q4_k.gguf 10240 0 99 0 2048 128,512 1,2,3,4 # bench the Q4_K model

./parallel -m ./models/openhermes-7b-v2.5/ggml-model-f16.gguf -n -1 -c 4096 --cont_batching --parallel 4 --sequences 64 --n-gpu-layers 99 -s 1

# start llama.cpp server, max 4 clients in parallel, prompt size 2048, max seq 512, listen on port 8888
LLAMA_CUBLAS=1 make -j server && ./server -m models/openhermes-7b-v2.5/ggml-model-q4_k.gguf --port 8888 --host 0.0.0.0 --ctx-size 10240 --parallel 4 -ngl 99 -n 512

# send a completion request via curl
curl -s http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
            },
            {
                "role": "user",
                "content": "Write a limerick about python exceptions"
            }
        ]
    }' | jq
