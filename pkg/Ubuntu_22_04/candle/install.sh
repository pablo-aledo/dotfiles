nvcc --version
nvidia-smi --query-gpu=compute_cap --format=csv

cargo new candle-myapp
cd candle-myapp

cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"
cargo build
