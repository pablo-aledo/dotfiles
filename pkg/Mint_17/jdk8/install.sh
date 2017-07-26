cd /tmp
wget --no-cookies --no-check-certificate --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com%2F; oraclelicense=accept-securebackup-cookie" "http://download.oracle.com/otn-pub/java/jdk/8u51-b16/jdk-8u51-linux-x64.tar.gz"
cd /opt
sudo tar xzf $OLDPWD/jdk-8u51-linux-x64.tar.gz
cd jdk1.8.0_51
sudo update-alternatives --install /usr/bin/java java /opt/jdk1.8.0_51/bin/java 2
sudo update-alternatives --config java
sudo update-alternatives --install /usr/bin/jar jar /opt/jdk1.8.0_51/bin/jar 2
sudo update-alternatives --install /usr/bin/javac javac /opt/jdk1.8.0_51/bin/javac 2
sudo update-alternatives --set jar /opt/jdk1.8.0_51/bin/jar
sudo update-alternatives --set javac /opt/jdk1.8.0_51/bin/javac 
echo 'export JAVA_HOME=/opt/jdk1.8.0_51' >> ~/.paths
echo 'export JRE_HOME=$JAVA_HOME/jre' >> ~/.paths
echo 'export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH' >> ~/.paths
