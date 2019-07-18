wget https://github.com/libevent/libevent/releases/download/release-2.0.22-stable/libevent-2.0.22-stable.tar.gz
tar xvf libevent-2.0.22-stable.tar.gz
cd libevent-2.0.22-stable
./configure --prefix=$HOME
make # use make -j 8 to speed it up if your machine is capable
make install

wget https://github.com/tmux/tmux/releases/download/2.2/tmux-2.2.tar.gz
tar xvf tmux-2.2
./configure --prefix=$HOME CFLAGS="-I$HOME/include" LDFLAGS="-L$HOME/lib"
make
make install
