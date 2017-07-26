
#!/bin/ash

ROOT=$(dirname $(readlink -f $0))/..
HOME=$(cd; pwd)

# adapt pkg
sed -i 's/REPOSITORY_FOLDER.*/REPOSITORY_FOLDER=\/dotfiles_master\/repository/g' $ROOT/source/.shell/pkg

# source pkg and install some stuff
. $ROOT/source/.shell/pkg
pkg install zsh
pkg install unzip
pkg install tmux

# rm files
rm $HOME/.tmux.conf
rm $HOME/.shell

# link files
ln -s $ROOT/minimal/.tmux.conf  $HOME/.tmux.conf
ln -s $ROOT/minimal/.shell $HOME/.shell

