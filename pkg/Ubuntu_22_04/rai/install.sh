wget https://github.com/Classevelabs/rai/releases/download/v0.2.5/rai-0.2.5-x86_64-unknown-linux-gnu.tar.gz
tar -xvzf rai-0.2.5-x86_64-unknown-linux-gnu.tar.gz
cd rai-0.2.5-x86_64-unknown-linux-gnu
#install -m 0755 rai ~/.local/bin/

pip install -U huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
./rai convert Qwen2.5-0.5B-Instruct/ -o qwen2.5-0.5b-instruct-q4.raimodel

./rai run qwen2.5-0.5b-instruct-q4.raimodel \
  --chat-template chatml \
  --prompt "Explain photosynthesis in simple terms." \
  --max-tokens 64
./rai serve qwen2.5-0.5b-instruct-q4.raimodel
