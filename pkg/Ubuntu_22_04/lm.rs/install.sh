cd
git clone https://github.com/samuel-vitorino/lm.rs
cd lm.rs
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin chat
curl -L 'https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-Q8_0-LMRS/resolve/main/tokenizer.bin?download=true' -o 'tokenizer.bin'
curl -L 'https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-Q8_0-LMRS/resolve/main/llama3.2-1b-it-q80.lmrs?download=true' -o 'llama3.2-1b-it-q80.lmrs'
./target/release/chat --model llama3.2-1b-it-q80.lmrs
