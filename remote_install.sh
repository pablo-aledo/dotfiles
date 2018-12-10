# wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/remote_install.sh -O - | bash

cd ~
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg -O .pkg

echo 'pkg update'        >> .pkg # requires superuser
echo 'pkg install zsh'   >> .pkg # requires superuser
echo 'pkg install git'   >> .pkg # requires superuser
echo 'pkg install unzip' >> .pkg # requires superuser
echo 'pkg install which' >> .pkg # requires superuser
echo 'pkg install sudo'  >> .pkg # requires superuser
echo 'pkg install grep'  >> .pkg # requires superuser
echo 'git clone --depth 1 https://github.com/pablo-aledo/dotfiles.git .dotfiles' >> .pkg
echo 'cd .dotfiles'      >> .pkg
echo 'touch  ~/.paths'      >> .pkg
echo 'source ~/.paths'      >> .pkg
echo 'source ./install.sh'  >> .pkg

source .pkg
