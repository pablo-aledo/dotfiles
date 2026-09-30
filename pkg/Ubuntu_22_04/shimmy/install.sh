wget https://github.com/Michael-A-Kuykendall/shimmy/releases/download/v2.6.1/shimmy-linux-x86_64
chmod +x ./shimmy-linux-x86_64

hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ./TinyLlama

./shimmy-linux-x86_64 serve --model-path ./TinyLlama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --bind 127.0.0.1:11435

shimmy list --short
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}'
