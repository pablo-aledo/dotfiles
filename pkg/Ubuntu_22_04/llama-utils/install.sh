pkg install wasmedge

mkd ~/mistral-7b-instruct
wget https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
curl -LO https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm

wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-chat.wasm -p mistral-instruct-v0.1
