sudo wget https://github.com/sqshq/sampler/releases/download/v1.0.3/sampler-1.0.3-linux-amd64 -O /usr/local/bin/sampler
sudo chmod +x /usr/local/bin/sampler
cd
wget https://raw.githubusercontent.com/sqshq/sampler/master/example.yml
sudo sampler --config example.yml
