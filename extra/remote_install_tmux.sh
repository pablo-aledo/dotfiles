# wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/remote_install.sh -O - | bash

cd ~
rm -rf .dotfiles_pga
mkdir  .dotfiles_pga
cd     .dotfiles_pga

wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg -O .pkg

echo 'git clone https://github.com/pablo-aledo/dotfiles.git .dotfiles' >> .pkg
echo 'sed -i s@^HOME=.*@HOME=$PWD@g .dotfiles/install.sh'              >> .pkg
echo 'cd .dotfiles'      >> .pkg
echo 'bash ./install.sh' >> .pkg
echo 'touch ~/.paths'      >> .pkg

bash .pkg

echo tttt >> .zshrc

cat << EOF >> alias
alias tmux_pga='HOME=%PWD% tmux'
alias  zsh_pga='HOME=%PWD% zsh'
alias  vim_pga='HOME=%PWD% vim'
EOF

sed -i s/%PWD%/$PWD/g alias
