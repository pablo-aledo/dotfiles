from centos:7

run yum install -y epel-release
run yum -y update

run yum install -y \
    autoconf \
    bison \
    byacc \
    flex \
    gcc-c++ \
    git \
    hiredis-devel \
    jsoncpp-devel \
    libarchive-devel \
    libevent-devel \
    libtool \
    libtool \
    lua-devel \
    lua-static \
    make \
    openblas-devel \
    openblas-static \
    openmpi-devel \
    openssl-devel \
    python-devel \
    qt5-qtbase-devel \
    sudo \
    unzip \
    mlocate \
    wget \
    ccache

# install boost-devel
run sudo mkdir /usr/src/boost_165; sudo chmod 777 /usr/src/boost_165; cd /usr/src/boost_165 && \
    wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz -O boost.tar.gz && \
    tar -xvzf boost.tar.gz && \
    cd boost_1_65_1 && \
    ./bootstrap.sh && \
    sudo ./b2 install

# install protobuf351
run sudo mkdir -p /usr/src/protobuf351; sudo chmod 777 /usr/src/protobuf351; cd /usr/src/protobuf351 && \
    wget https://github.com/google/protobuf/archive/v3.5.1.tar.gz && \
    tar -xvzf v3.5.1.tar.gz && \
    cd protobuf-3.5.1 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    sudo make install && \
    sudo ldconfig

# install thrift-devel_092
run sudo ln -s /usr/lib64/liblua.so /usr/lib64/liblua5.2.so && \
    sudo ln -s /usr/src/thrift092/thrift-0.9.2/lib/lua/.libs/liblualongnumber.so /usr/lib/liblualongnumber.so && \
    sudo mkdir /usr/src/thrift092/; sudo chmod 777 /usr/src/thrift092; cd /usr/src/thrift092 && \
    wget https://github.com/apache/thrift/archive/0.9.2.tar.gz && \
    tar -xvzf 0.9.2.tar.gz && \
    cd thrift-0.9.2 && \
    ./bootstrap.sh && \
    ./configure && \
    make && \
    sudo make install

run echo 'export PATH="/usr/lib64/ccache:$PATH"' >> /root/.bashrc
