# ncurses
wget http://ftp.gnu.org/pub/gnu/ncurses/ncurses-5.9.tar.gz
tar xvzf ncurses-5.9.tar.gz
cd ncurses-5.9
./configure --prefix=$HOME/local
make -j8
make install
cd ..

# libevent
git clone git://github.com/libevent/libevent.git -b release-2.0.21-stable
cd libevent
./autogen.sh
./configure --prefix=$HOME/local
make -j8
make install
cd ..

# tmux
git clone git://git.code.sf.net/p/tmux/tmux-code -b 1.8
cd tmux
./configure --prefix=$HOME/local CPPFLAGS="-I$HOME/local/include -I$HOME/local/include/ncurses" LDFLAGS="-static -L$HOME/local/include -L$HOME/local/include/ncurses -L$HOME/local/lib"
make -j8
make install