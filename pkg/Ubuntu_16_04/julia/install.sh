cd /tmp/
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.3-linux-x86_64.tar.gz
tar -xvzf julia-1.6.3-linux-x86_64.tar.gz
sudo mv julia-1.6.3 /opt/
sudo ln -s /opt/julia-1.6.3/bin/julia /usr/local/bin/julia