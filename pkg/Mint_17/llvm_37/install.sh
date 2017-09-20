sudo apt-get install -y subversion cmake g++ python2.7

path /usr/bin 
sudo ln -s /usr/bin/python2.7 /usr/bin/python

mksrcdir /usr/share/llvm-3.7
mksrcdir /usr/src/llvm-3.7
cd /usr/src/llvm-3.7

svn co http://llvm.org/svn/llvm-project/llvm/tags/RELEASE_371/final /usr/src/llvm-3.7
svn co http://llvm.org/svn/llvm-project/cfe/tags/RELEASE_371/final /usr/src/llvm-3.7/tools/clang
svn co http://llvm.org/svn/llvm-project/libcxx/tags/RELEASE_371/final /usr/src/llvm-3.7/projects/libcxx
svn co http://llvm.org/svn/llvm-project/libcxxabi/tags/RELEASE_371/final /usr/src/llvm-3.7/projects/libcxxabi

mkd build
../configure --prefix=/usr/share/llvm-3.7
make -j `nproc`
sudo make install

