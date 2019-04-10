cd
wget https://cmake.org/files/v3.4/cmake-3.4.1-Linux-x86_64.tar.gz
tar xf cmake-3.4.1-Linux-x86_64.tar.gz
export PATH="`pwd`/cmake-3.4.1-Linux-x86_64/bin:$PATH" # save it in .bashrc if needed
echo 'export PATH="`pwd`/cmake-3.4.1-Linux-x86_64/bin:$PATH"' >> ~/.paths
