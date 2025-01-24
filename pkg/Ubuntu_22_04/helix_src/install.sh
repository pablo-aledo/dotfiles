pkg install cargo

cd
git clone https://github.com/helix-editor/helix.git
cd helix
cargo build --release

sudo mkdir /usr/lib/helix
sudo cp target/release/hx /usr/lib/helix/hx
sudo cp -r runtime /usr/lib/helix/
sudo ln -s /usr/lib/helix/hx /usr/bin/hx
