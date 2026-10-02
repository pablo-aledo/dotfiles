git clone https://github.com/cool-japan/oxillama
cd oxillama
cargo build --release

hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ./TinyLlama

./target/release/oxillama run \
  --model ./TinyLlama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256 \
  --temp 0.7

./target/release/oxillama chat \
  --model ./TinyLlama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

./target/release/oxillama serve \
  --model ./TinyLlama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --admin-token hola

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}'
