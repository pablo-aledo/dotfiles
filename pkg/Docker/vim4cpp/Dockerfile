FROM debian:stretch
MAINTAINER Wang Boyu <boyu.wby@gmail.com>

RUN apt-get update && apt-get install -y \
    cmake \
    exuberant-ctags \
    git \
    libcanberra-gtk-module \
    libgtk2.0-dev \
    libncurses5-dev \
    libperl-dev \
    libx11-dev \
    libxt-dev \
    lua5.1 \
    lua5.1-dev \
    wget \
    python-dev \
    ruby-dev \
  && rm -rf /var/lib/apt/lists/*

RUN cd /home \
  && git clone https://github.com/vim/vim.git \
  && cd vim \
  && ./configure --with-features=huge --enable-pythoninterp \
    --enable-rubyinterp --enable-luainterp \
    --enable-perlinterp --enable-gui=gtk2 \
    --enable-cscope --prefix=/usr \
    --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu \
  && make -j4 \
  && make install \
  && cd .. \
  && rm -rf vim

RUN wget -qO- \
    https://raw.githubusercontent.com/yakumioto/YaHei-Consolas-Hybrid-1.12/master/install.sh | sh

COPY ./resources/ /home/root/

RUN git clone https://github.com/VundleVim/Vundle.vim.git /root/.vim/bundle/Vundle.vim \
  && vim -S /home/root/plugin-list +PluginInstall +qall

RUN rm /home/root/plugin-list \
  && mv /home/root/dot-vimrc /root/.vimrc \
  && mv /home/root/dot-indexer_files /root/.indexer_files \
  && mv /home/root/dot-ycm_extra_conf.py /root/.vim/.ycm_extra_conf.py \
  && mkdir -p /root/.vim/bundle/ultisnips/mysnippets \
  && mv /home/root/cpp.snippets /root/.vim/bundle/ultisnips/mysnippets \
  && cd /usr/include/c++/6 \
  && ctags -R --c++-kinds=+l+x+p --fields=+iaSl --extra=+q --language-force=c++ -f stdcpp.tags \
  && mkdir -p /root/.vim/tags \
  && mv stdcpp.tags /root/.vim/tags

RUN cd /home/root \
  && wget http://releases.llvm.org/3.9.1/clang+llvm-3.9.1-x86_64-linux-gnu-debian8.tar.xz \
  && tar -xvf clang+llvm-3.9.1-x86_64-linux-gnu-debian8.tar.xz \
  && mkdir -p ycm_build \
  && cd ycm_build \
  && cmake -G "Unix Makefiles" \
    -DPATH_TO_LLVM_ROOT=/home/root/clang+llvm-3.9.1-x86_64-linux-gnu-debian8 \
    . \
    /root/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp \
  && cmake --build . --config Release --target ycm_core -- -j4

RUN cd /home/root \
  && cp -R clang+llvm-3.9.1-x86_64-linux-gnu-debian8/* /usr/local/ \
  && cp /usr/local/lib/libc++*.so* /usr/lib/ \
  && sed -i "$ a export CC=/usr/local/bin/clang\nexport CXX=/usr/local/bin/clang++" /root/.bashrc \
  && mkdir -p /usr/include/c++/v1/ \
  && cp -R /usr/local/include/c++/v1/* /usr/include/c++/v1/ \
  && rm -rf clang+llvm-3.9.1-x86_64-linux-gnu-debian8/ \
    clang+llvm-3.9.1-x86_64-linux-gnu-debian8.tar.xz \
    ycm_build

CMD ["gvim", "-f"]
