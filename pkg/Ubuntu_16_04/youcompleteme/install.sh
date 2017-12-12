git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

cp vimrc ~/.vimrc

vim +PluginInstall +qall

sudo apt-get install build-essential cmake
sudo apt-get install python-dev python3-dev

cd ~/.vim/bundle/YouCompleteMe
./install.py --clang-completer





