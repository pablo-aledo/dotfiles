ROOT=$(dirname $(readlink -f $0))
HOME=$(cd; pwd)

escape(){
	echo $1 | sed -e "s/\//\\\\\//g"
}

restore(){
	echo -e "\e[34m Restore file ... \e[0m" `basename $1`
	src=$(echo $1 | sed "s/`escape $HOME`/`escape $HOME`\/.dotfiles_bak/g")
	dst=$1
	mkdir -p $(dirname $dst)
	\cp -f $src $dst
	[ -e $src ] || rm -f $dst 2>/dev/null
}

run(){
	echo -e "\e[34m Sourcing file ... \e[0m" `basename $1`
	cd $ROOT/uninstall
	source $1
}

# uninstall files
for a in `find $ROOT/uninstall -type f 2>/dev/null | sort -g`
do
	run $a
done

# link files
for a in `find $ROOT/link -type f 2>/dev/null`
do
	restore $(echo $a | sed "s/`escape $ROOT`\/link/`escape $HOME`/g")
done

# copy files
for a in `find $ROOT/copy -type f 2>/dev/null`
do
	restore $(echo $a | sed "s/`escape $ROOT`\/copy/`escape $HOME`/g")
done

# source files
for a in `find $ROOT/source -type d 2>/dev/null | egrep -v source$ | sort -g`
do
	restore $(echo $a | sed "s/`escape $ROOT`\/source/`escape $HOME`/g")
done

