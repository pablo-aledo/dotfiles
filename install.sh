#!/bin/bash

ROOT=$(dirname $(readlink -f -- $0))
HOME=$(cd; pwd)

backup(){
	echo -e "\e[34m Backup file ... \e[0m" `basename $1`
	src=$1
	dst=$(echo $1 | sed "s/`escape $HOME`/`escape $HOME`\/.dotfiles_bak/g")
	mkdir -p $(dirname $dst)
	[ -e $dst ] || mv $src $dst 2>/dev/null
}

escape(){
	echo $1 | sed -e "s/\//\\\\\//g"
}

link(){
	echo -e "\e[34m linking file ... \e[0m" `basename $1`
	src=$1
	dst=$(echo $1 | sed "s/`escape $ROOT`\/link/`escape $HOME`/g")
	mkdir -p $(dirname $dst)
	ln -sf $src $dst
}

link_if_new(){
	src=$1
	dst=$(echo $1 | sed "s/`escape $ROOT`\/link_if_new/`escape $HOME`/g")
	echo -e "\e[34m Checking file ... \e[0m $dst"
	[ -e $dst ] && return
	echo -e "\e[34m linking file ... \e[0m" `basename $1`
	mkdir -p $(dirname $dst)
	ln -sf $src $dst
}

copy(){
	echo -e "\e[34m copying file ... \e[0m" `basename $1`
	src=$1
	dst=$(echo $1 | sed "s/`escape $ROOT`\/copy/`escape $HOME`/g")
	mkdir -p $(dirname $dst)
	cp $src $dst
}

sourcefile(){
	echo -e "\e[34m Adding as source file ... \e[0m" `basename $1`
	for a in $1/*
	do
		src=$a
		dst=$(dirname $(echo $src | sed "s/`escape $ROOT`\/source/`escape $HOME`/g"))
		[ "`grep "$src" "$dst" 2>/dev/null`" ] || echo source $src >> $dst
	done
}

sourcecontent(){
	echo -e "\e[34m Adding content of file ... \e[0m" `basename $1`
	for a in $1/*
	do
		src=$a
		dst=$(dirname $(echo $src | sed "s/`escape $ROOT`\/source_content/`escape $HOME`/g"))
		mkdir -p $(dirname $dst)
		cat $src >> $dst
	done
}

run(){
	echo -e "\e[34m Sourcing file ... \e[0m" `basename $1`
	cd $ROOT/run
	source $1
}

# exec files
for a in `find $ROOT/run -type f 2>/dev/null | sort -g`
do
	run $a
done

# link files
for a in `find $ROOT/link -type f 2>/dev/null`
do
	backup $(echo $a | sed "s/`escape $ROOT`\/link/`escape $HOME`/g")
	link $a
done

# link_new files
for a in `find $ROOT/link_if_new -type f 2>/dev/null`
do
	link_if_new $a
done

# copy files
for a in `find $ROOT/copy -type f 2>/dev/null`
do
	backup $(echo $a | sed "s/`escape $ROOT`\/copy/`escape $HOME`/g")
	copy $a
done

# source files
for a in `find $ROOT/source -type d 2>/dev/null | egrep -v source$ | sort -g`
do
	backup $(echo $a | sed "s/`escape $ROOT`\/source/`escape $HOME`/g")
	sourcefile $a
done

# source content
for a in `find $ROOT/source_content -type f 2>/dev/null | egrep -v source_content$ | sort -g | xargs -l1 dirname | uniq`
do 
	backup $(echo $a | sed "s/`escape $ROOT`\/source_content/`escape $HOME`/g")
	rm -f $(echo $a | sed "s/`escape $ROOT`\/source_content/`escape $HOME`/g")
	sourcecontent $a
done

# final message
echo "\e[32m There's no place like \e[33m $(wget http://ipinfo.io/ip -qO -) \e[0m"
