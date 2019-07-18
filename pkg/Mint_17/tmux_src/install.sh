ROOTDIR=$OLDPWD

wget https://github.com/libevent/libevent/releases/download/release-2.0.22-stable/libevent-2.0.22-stable.tar.gz
tar xvf libevent-2.0.22-stable.tar.gz
cd libevent-2.0.22-stable
./configure --prefix=$ROOTDIR
make # use make -j 8 to speed it up if your machine is capable
make install
cd ..

wget https://ftp.gnu.org/gnu/ncurses/ncurses-5.7.tar.gz
tar -xvzf ncurses-5.7.tar.gz
cd ncureses-5.7
./configure --prefix=$ROOTDIR
make
make install
cd ..

wget https://github.com/tmux/tmux/releases/download/2.2/tmux-2.2.tar.gz
tar xvf tmux-2.2
./configure --prefix=$ROOTDIR CFLAGS="-I$ROOTDIR/include" LDFLAGS="-L$ROOTDIR/lib"
make
make install
