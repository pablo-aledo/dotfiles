CRANE_DIR="${CRANE_DIR:-$HOME/crane}"
CRANE_REPO="https://github.com/lucasjinreal/Crane.git"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
TEST_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
TEST_MODEL_DIR="$CRANE_DIR/checkpoints/Qwen2.5-0.5B-Instruct"

sudo apt-get update -y
sudo apt-get install -y \
  build-essential \
  pkg-config \
  cmake \
  git \
  curl \
  wget \
  libssl-dev \
  libclang-dev \
  clang \
  ca-certificates \
  python3 \
  python3-pip \
  python3-venv
  ffmpeg \
  unzip

git clone "$CRANE_REPO" "$CRANE_DIR"
cd "$CRANE_DIR"
cargo build --release -p crane -p crane-serve -p crane-examples

hf download "$TEST_MODEL_ID" --local-dir "$TEST_MODEL_DIR"

# ./target/release/chat_cli -m checkpoints/Qwen2.5-0.5B-Instruct --model-type qwen25
# ./target/release/chat_streaming -m checkpoints/Qwen2.5-0.5B-Instruct
# ./target/release/crane-serve \
#   --model-path checkpoints/Qwen2.5-0.5B-Instruct \
#   --model-type qwen25 \
#   --port 8080
#
# curl http://127.0.0.1:8080/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#         "model": "Qwen2.5-0.5B-Instruct",
#         "messages": [{"role": "user", "content": "Cuenta hasta 5"}],
#         "stream": true
#       }'
