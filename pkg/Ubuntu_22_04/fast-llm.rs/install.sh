INSTALL_DIR="${HOME}/fast-llm.rs"
REPO_URL="https://github.com/Vaibhavs10/fast-llm.rs"

sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    cmake \
    ca-certificates

#curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#source "${HOME}/.cargo/env"

git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

sed -i 's/hf-hub = { version = "0.3.2", features = \["tokio"\] }/hf-hub = { version = "0.4", features = ["tokio"] }/' Cargo.toml
sed -i 's/tokenizers = { version = "0.15.0", default-features = false, features=\["onig"\] }/tokenizers = { version = "0.21", default-features = false, features=["onig"] }/' Cargo.toml

cargo build --release

./target/release/fast-llm \
    --which 7b-mistral-instruct-v0.2 \
    --prompt "¿Cuál es el sentido de la vida según un perro?" \
    --sample-len 100
