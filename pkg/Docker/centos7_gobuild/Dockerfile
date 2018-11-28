from centos:7

run yum install -y \
    bash \
    git \
    curl \
    gcc \
    wget

run wget https://storage.googleapis.com/golang/go1.9.2.linux-amd64.tar.gz -O - | tar -zxv -C /usr/local/ && \
    mkdir ~/go && \
    echo 'export GOPATH=~/go' >> ~/bashrc && \
    echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc && \
    echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.bashrc

env GOPATH=/root/go
env PATH=/usr/local/go/bin:/root/go/bin:/usr/bin
run mkdir -p /root/go/bin

run curl https://raw.githubusercontent.com/golang/dep/master/install.sh | sh
run cp /root/go/bin/dep /bin/

COPY       build.sh /bin
RUN        chmod +x /bin/build.sh

ENTRYPOINT     ["build.sh"]
