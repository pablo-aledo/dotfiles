# apt
sudo apt-get install -y nvidia-opencl-dev ocl-icd-libopencl1 opencl-headers clinfo

# compile
mkdir -p "$HOME/tmp/tar_gz"
cd "$HOME/tmp/tar_gz"
if [[ ! -f "$HOME/tmp/tar_gz/hashcat-4.0.1.tar.gz" ]]; then
  wget -O "$HOME/tmp/tar_gz/hashcat-4.0.1.tar.gz" "https://github.com/hashcat/hashcat/archive/v4.0.1.tar.gz"
fi
tar xf "$HOME/tmp/tar_gz/hashcat-4.0.1.tar.gz"
cd "$HOME/tmp/tar_gz/hashcat-4.0.1"
make clean
make distclean
make

# install
sudo make install
sudo ldconfig
