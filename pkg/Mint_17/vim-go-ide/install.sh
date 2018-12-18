mkdir -p ~/.vim/autoload ~/.vim/bundle
curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim
git clone http://github.com/farazdagi/vim-go-ide.git ~/.vim_go_runtime
sh ~/.vim_go_runtime/bin/install
vim -u ~/.vimrc.go +GoInstallBinaries

cat << EOF > ~/.profile
export GOPATH=$HOME/go
export GOROOT=/usr/local/opt/go/libexec
export PATH=$PATH:$GOPATH/bin
export PATH=$PATH:$GOROOT/bin
alias vimgo='vim -u ~/.vimrc.go'
EOF

cat << EOF > ~/.bashrc
source ~/.profile
EOF

cat << EOF > ~/.zshrc
[[ -e ~/.profile ]] && emulate sh -c 'source ~/.profile'
EOF

cat << EOF > ~/.bash-profile
if [ -f ~/.bashrc ]; then
   source ~/.bashrc
fi
EOF

