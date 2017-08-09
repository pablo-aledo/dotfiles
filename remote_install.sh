# wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/remote_install.sh -O - | bash

cd ~
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg -O .pkg

echo 'pkg update'        >> .pkg
echo 'pkg install zsh'   >> .pkg
echo 'pkg install git'   >> .pkg
echo 'pkg install unzip' >> .pkg
echo 'git clone https://github.com/pablo-aledo/dotfiles.git .dotfiles' >> .pkg
echo 'cd .dotfiles'      >> .pkg
echo 'bash ./install.sh' >> .pkg
echo 'touch ~/.paths'      >> .pkg

bash .pkg
