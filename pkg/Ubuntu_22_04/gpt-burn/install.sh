sudo apt install -y pkg-config libssl-dev libvulkan1 mesa-vulkan-drivers vulkan-tools
cargo run --release
curl -s 'https://drive.usercontent.google.com/download?id=1GGLaPnmPQ8Z2B9vJQoI6-K128X9LJKG0&export=download&confirm=t' | tar xzf -
gpt-burn run ./model_83M
