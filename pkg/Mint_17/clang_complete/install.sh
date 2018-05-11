git clone https://github.com/Rip-Rip/clang_complete.git /tmp/clang_complete
cd /tmp/clang_complete
make install

cat << EOF >> ~/.vimrc 
" path to directory where library can be found
let g:clang_library_path='/usr/lib/llvm-3.8/lib'
" or path directly to the library file
" let g:clang_library_path='/usr/lib64/libclang.so.3.8'
EOF

cat << EOF >> /tmp/.clang_complete
-DDEBUG
-include ../config.h
-I../common
-I/usr/include/c++/4.5.3/
-I/usr/include/c++/4.5.3/x86_64-slackware-linux/
EOF

