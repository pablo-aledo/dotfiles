pkg install cargo

cd
git clone https://github.com/helix-editor/helix.git
cd helix
cargo build --release

sudo mv target/release/hx /usr/bin/hx
