git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

cp vimrc ~/.vimrc

vim +PluginInstall +qall

sudo apt-get install -y build-essential cmake
sudo apt-get install -y python-dev python3-dev

cd ~/.vim/bundle/YouCompleteMe
./install.py --clang-completer





