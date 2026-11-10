cd /tmp
wget https://github.com/cvlmtg/HUME/releases/download/v0.19.0/hume-0.19.0-9325f2b6c-x86_64-unknown-linux-gnu.tar.gz
tar -xvzf hume-0.19.0-9325f2b6c-x86_64-unknown-linux-gnu.tar.gz
mkdir -p ~/.local && cp -R hume-*/bin hume-*/share ~/.local/
