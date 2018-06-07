(cd /tmp && bak ~/.vimrc ~/.vim && rm -rf ~/.vimrc ~/.vim )

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

echo source $PWD/vundle > ~/.vimrc

for a in $(find $PWD/../../../source/.vimrc/ -type f | sort)
do
	echo source $a >> ~/.vimrc
done

vim +PluginInstall +qall

mkdir -p ~/.vim
cd ~/.vim
for a in $(find $OLDPWD -name '*.zip')
do
	unzip -o $a >/dev/null 2>/dev/null
done

cd -

ROOT=$HOME/.dotfiles

for a in `find $PWD/../../../link/.vim -type f 2>/dev/null`
do
	src=$a
	dst=$(echo $a | sed "s@$PWD/../../../link/@$HOME/@g")
	mkdir -p $(dirname $dst)
	ln -sf $src $dst
done
