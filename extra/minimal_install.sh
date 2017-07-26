#!/bin/zsh

ROOT=$(dirname $(readlink -f $0))/..
HOME=$(cd; pwd)

# rm files
rm $HOME/.tmux.conf
rm $HOME/.profile
rm $HOME/.vimrc
rm $HOME/.shell

# link files
ln -s $ROOT/minimal/.tmux.conf  $HOME/.tmux.conf
ln -s $ROOT/minimal/.profile $HOME/.profile

# source files
echo source $ROOT/source/.vimrc/00_basic  >  $HOME/.vimrc
echo source $ROOT/source/.vimrc/07_maps   >> $HOME/.vimrc
echo source $ROOT/source/.vimrc/12_search >> $HOME/.vimrc

echo source $ROOT/source/.shell/avaxhome  >  $HOME/.shell
echo source $ROOT/source/.shell/mkd       >> $HOME/.shell
echo source $ROOT/source/.shell/exit      >> $HOME/.shell

# Change files
sed -i 's/<C-F3>/F/g' $ROOT/source/.vimrc/12_search 2>/dev/null
sed -i 's/.media.DATA.ebooks/~/g' $ROOT/source/.shell/avaxhome 2>/dev/null
