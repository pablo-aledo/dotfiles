# wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/remote_install.sh -O - | bash

cd ~
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg -O .pkg

echo 'pkg update'        >> .pkg # requires superuser
echo 'pkg install zsh'   >> .pkg # requires superuser
echo 'pkg install git'   >> .pkg # requires superuser
echo 'pkg install unzip' >> .pkg # requires superuser
echo 'git clone https://github.com/pablo-aledo/dotfiles.git .dotfiles' >> .pkg
echo 'cd .dotfiles'      >> .pkg
echo 'bash ./install.sh' >> .pkg
echo 'touch ~/.paths'      >> .pkg

bash .pkg
