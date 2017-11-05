ROOT=$(dirname $(readlink -f $0))/..
HOME=$(cd; pwd)
vimfolder=$ROOT/source/.vimrc

for a in $vimfolder/*
do
	src=$a
	dst=~/vimrc
	echo source $src >> $dst
done

alias vim='vim +":so ~/vimrc"'
