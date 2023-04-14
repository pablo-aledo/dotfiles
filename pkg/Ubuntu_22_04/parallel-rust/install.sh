cd /tmp
git clone https://github.com/mmstick/parallel.git
cd parallel
cargo build
sudo mv target/debug/parallel /usr/bin/parallel
