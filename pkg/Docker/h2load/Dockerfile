FROM   ubuntu

ENV    LD_LIBRARY_PATH=/usr/local/lib/

RUN    apt-get update && \
       apt install       \
       g++               \ 
       make              \ 
       binutils          \ 
       autoconf          \ 
       automake          \ 
       autotools-dev     \ 
       libtool           \ 
       pkg-config        \ 
       zlib1g-dev        \ 
       libcunit1-dev     \ 
       libssl-dev        \ 
       libxml2-dev       \ 
       libev-dev         \ 
       libevent-dev      \ 
       git               \ 
       python            \ 
       python-dev        \ 
       libjemalloc-dev   \ 
       libjansson-dev    \ 
       libsystemd-dev    \ 
       libc-ares-dev     \ 
       -y --force-yes && \
       git clone https://github.com/nghttp2/nghttp2.git && cd nghttp2 && \
       autoreconf -i  && \
       automake       && \
       autoconf       && \
       ./configure --enable-app && \
       make           && \
       make install
