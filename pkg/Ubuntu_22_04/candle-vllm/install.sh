REPO_URL="https://github.com/EricLBuehler/candle-vllm.git"
REPO_DIR="${HOME}/candle-vllm"

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  clang \
  libclang-dev \
  cmake \
  make \
  pkg-config \
  perl \
  build-essential \
  libssl-dev \
  numactl

git clone "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "${HOME}/.cargo/env"

cargo build --release --features cuda

# candle-vllm --m unsloth/Qwen3-4B-Instruct-2507-GGUF \
#              --f Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ui-server
