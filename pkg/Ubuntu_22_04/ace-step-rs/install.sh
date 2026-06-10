sed -i 's|{ path = "/home/marenz/repos/candle/candle-core" }|{ git = "https://github.com/Marenz/candle", branch = "fast-conv-transpose1d-no-cudnn" }|g' Cargo.toml
sed -i 's|{ path = "/home/marenz/repos/candle/candle-nn" }|{ git = "https://github.com/Marenz/candle", branch = "fast-conv-transpose1d-no-cudnn" }|g' Cargo.toml
sed -i 's|{ path = "/home/marenz/repos/candle/candle-transformers" }|{ git = "https://github.com/Marenz/candle", branch = "fast-conv-transpose1d-no-cudnn" }|g' Cargo.toml

cargo build --no-default-features
cargo build --no-default-features --bin ace-step

./target/debug/ace-step \
  --caption "upbeat jazz with piano and drums, bpm: 120, key: C major" \
  --lyrics "[verse]
Walking down the street on a sunny day" \
  --duration 30 \
  --output /tmp/test.wav
