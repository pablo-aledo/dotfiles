# http://www.madox.net/blog/2008/11/26/compiling-a-toolchain-for-arm7-under-ubuntu/
sudo apt-get install -y libgmp3-dev libmpfr-dev texinfo g++

cd ~
mkdir arm-elf arm-elf/src

cd ~/arm-elf/src
wget https://ftp.gnu.org/gnu/binutils/binutils-2.19.1.tar.bz2
wget https://ftp.gnu.org/gnu/gcc/gcc-4.3.2/gcc-4.3.2.tar.bz2
wget http://pkgs.fedoraproject.org/repo/pkgs/xen/newlib-1.16.0.tar.gz/bf8f1f9e3ca83d732c00a79a6ef29bc4/newlib-1.16.0.tar.gz
wget http://pkgs.fedoraproject.org/repo/pkgs/insight/insight-6.8.tar.bz2/b403972b35520399663c7054e8132ca9/insight-6.8.tar.bz2

cd ~/arm-elf/src
tar -xvjf binutils-2.19.1.tar.bz2
tar -xvjf gcc-4.3.2.tar.bz2
tar -xvzf newlib-1.16.0.tar.gz
tar -xvjf insight-6.8.tar.bz2

cd ~/arm-elf/
mkdir build build/binutils-2.19 build/insight-6.8 build/gcc-4.3.2 build/newlib-1.16.0

cd ~/arm-elf/build/binutils-2.19
~/arm-elf/src/binutils-2.19.1/configure -target=arm-elf -prefix=/usr/local -enable-interwork -enable-multilib -with-float=soft -disable-werror
sudo make all install

cd ~/arm-elf/build/gcc-4.3.2
sudo mv /usr/bin/makeinfo /usr/bin/makeinfo.bak
sudo ln -s /bin/true /usr/bin/makeinfo
~/arm-elf/src/gcc-4.3.2/configure -target=arm-elf -prefix=/usr/local -enable-interwork -enable-multilib -with-float=soft -disable-werror -enable-languages="c,c++" -with-newlib  -with-headers=~/arm-elf/src/newlib-1.16.0/newlib/libc/include
sudo make all-gcc install-gcc

cd ~/arm-elf/build/newlib-1.16.0
sudo mv /usr/bin/makeinfo.bak /usr/bin/makeinfo
~/arm-elf/src/newlib-1.16.0/configure -target=arm-elf -prefix=/usr/local -enable-interwork -enable-multilib -with-float=soft -disable-werror
sudo make all install

cd ~/arm-elf/build/gcc-4.3.2
sudo mv /usr/bin/makeinfo /usr/bin/makeinfo.bak
sudo ln -s /bin/true /usr/bin/makeinfo
sudo make all install

cd ~/arm-elf/build/insight-6.8
~/arm-elf/src/insight-6.8/configure -target=arm-elf -prefix=/usr/local -enable-interwork -enable-multilib -with-float=soft -disable-werror
sudo make all install

cd /usr/local/bin/
ls arm*
