cd
wget https://cmake.org/files/v3.15/cmake-3.15.1-Linux-x86_64.tar.gz
tar xf cmake-3.15.1-Linux-x86_64.tar.gz
export PATH="$HOME/cmake-3.15.1-Linux-x86_64/bin:$PATH" # save it in .bashrc if needed
echo 'export PATH="~/cmake-3.15.1-Linux-x86_64/bin:$PATH"' | sed "s|~|$HOME|g" >> ~/.paths
