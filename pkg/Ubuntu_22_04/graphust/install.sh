cd
git clone https://github.com/crimory/graphust.git
cd graphust
cargo build --release
sudo cp target/release/graphust /usr/bin
# wget https://github.com/crimory/graphust/releases/download/v0.2.0/graphust.tar.gz -O - | sudo tar -xvz -C /usr/bin
# sudo chmod +x /usr/bin/graphust
