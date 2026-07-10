#cd /tmp/
#wget https://github.com/linuxrebel/DocuBrowser/releases/download/v0.9.1/docubrowser-foss_0.9.1-1_all.deb
#sudo apt install ./docubrowser-foss_0.9.0-7_all.deb

sudo tee /usr/local/bin/xdg-terminal-exec >/dev/null <<'EOF'
#!/bin/sh
exec x-terminal-emulator "$@"
EOF
sudo chmod +x /usr/local/bin/xdg-terminal-exec

cd /tmp/
wget https://github.com/linuxrebel/DocuBrowser/releases/download/v0.9.1/docubrowser-foss-0.9.1-1.tar.gz
tar -xvzf docubrowser-foss-0.9.1-1.tar.gz
cd docubrowser-foss-0.9.1-1
sudo ./install.sh
