cd /tmp
wget https://github.com/neovim/neovim/releases/download/nightly/nvim.appimage
chmod +x nvim.appimage
sudo mv nvim.appimage /usr/bin/nvim
git clone https://github.com/NvChad/NvChad ~/.config/nvim --depth 1
