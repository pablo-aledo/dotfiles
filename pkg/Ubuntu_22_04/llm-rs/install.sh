curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rustformers/llm/releases/download/v0.1.1/llm-cli-installer.sh | sh

cd
mkdir redpajama
cd redpajama
wget https://huggingface.co/rustformers/redpajama-3b-ggml/resolve/main/RedPajama-INCITE-Base-3B-v1-q4_0.bin

llm infer -a gptneox -m RedPajama-INCITE-Base-3B-v1-q4_0.bin -p "Rust is a cool programming language because" -r togethercomputer/RedPajama-INCITE-Base-3B-v1

