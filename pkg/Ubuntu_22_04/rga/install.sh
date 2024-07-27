sudo apt install -y ripgrep pandoc poppler-utils ffmpeg

cd /tmp
wget https://github.com/phiresky/ripgrep-all/releases/download/v0.10.6/ripgrep_all-v0.10.6-x86_64-unknown-linux-musl.tar.gz
tar -xvzf ripgrep_all-v0.10.6-x86_64-unknown-linux-musl.tar.gz
sudo mv ripgrep_all-v0.10.6-x86_64-unknown-linux-musl/rga-preproc /usr/bin
sudo mv ripgrep_all-v0.10.6-x86_64-unknown-linux-musl/rga-fzf /usr/bin
sudo mv ripgrep_all-v0.10.6-x86_64-unknown-linux-musl/rga /usr/bin

wget https://github.com/jgm/pandoc/releases/download/3.2.1/pandoc-3.2.1-linux-amd64.tar.gz
tar -xvzf pandoc-3.2.1-linux-amd64.tar.gz
sudo mv pandoc-3.2.1/bin/pandoc /usr/bin
sudo mv pandoc-3.2.1/bin/pandoc-lua /usr/bin
sudo mv pandoc-3.2.1/bin/pandoc-server /usr/bin
