pkg install wasmedge

mkd ~/mistral-7b-instruct
wget https://huggingface.co/second-state/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
curl -LO https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm
wasmedge --dir .:. --nn-preload default:GGML:AUTO:mistral-7b-instruct-v0.1.Q5_K_M.gguf llama-chat.wasm -p mistral-instruct-v0.1

#mkd ~/orca-2-13b
#wget https://huggingface.co/second-state/Orca-2-13B-GGUF/resolve/main/Orca-2-13b-ggml-model-q4_0.gguf
#curl -LO https://github.com/second-state/llama-utils/raw/main/chat/llama-chat.wasm
#wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13b-ggml-model-q4_0.gguf llama-chat.wasm -p chatml -s 'You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.' --stream-stdout

#curl -LO https://github.com/second-state/llama-utils/raw/main/api-server/llama-api-server.wasm
#wasmedge --dir .:. --nn-preload default:GGML:AUTO:Orca-2-13B.Q5_K_M.gguf llama-api-server.wasm -p chatml
#curl -X POST http://0.0.0.0:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content":"You are a helpful AI assistant"}, {"role":"user", "content":"What is the capital of France?"}], "model":"Orca-2-13B"}'

