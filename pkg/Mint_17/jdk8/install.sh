cd /tmp
wget http://monalisa.cern.ch/MONALISA/download/java/jdk-8u72-linux-x64.tar.gz
cd /opt
sudo tar xzf $OLDPWD/jdk-8u72-linux-x64.tar.gz
cd jdk1.8.0_72
sudo update-alternatives --install /usr/bin/java java /opt/jdk1.8.0_72/bin/java 2
sudo update-alternatives --install /usr/bin/jar jar /opt/jdk1.8.0_72/bin/jar 2
sudo update-alternatives --install /usr/bin/javac javac /opt/jdk1.8.0_72/bin/javac 2
sudo update-alternatives --set jar /opt/jdk1.8.0_72/bin/jar
sudo update-alternatives --set javac /opt/jdk1.8.0_72/bin/javac 
echo 'export JAVA_HOME=/opt/jdk1.8.0_72' >> ~/.paths
echo 'export JRE_HOME=$JAVA_HOME/jre' >> ~/.paths
echo 'export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH' >> ~/.paths
