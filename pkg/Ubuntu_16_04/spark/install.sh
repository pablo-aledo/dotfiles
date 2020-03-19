sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get install -y openjdk-8-jdk
sudo apt-get install -y openssh-server

sudo addgroup sparkgroup
sudo adduser --ingroup sparkgroup sparkuser

ssh-keygen -f ~/.ssh/id_rsa -t rsa -N ''
sudo mkdir /home/sparkuser/.ssh/
cat ~/.ssh/id_rsa.pub | sudo tee -a /home/sparkuser/.ssh/authorized_keys

echo 'sparkuser ALL=(ALL) ALL' | sudo tee -a /etc/sudoers

sudo mkdir /usr/local/spark
sudo chown -R sparkuser /usr/local/spark
sudo chmod -R 755 /usr/local/spark

sudo mkdir /usr/local/scala
sudo chown -R sparkuser /usr/local/scala
sudo chmod -R 755 /usr/local/scala

sudo mkdir -p /app/spark/tmp
sudo chown -R sparkuser /app/spark/tmp
sudo chmod -R 755 /app/spark/tmp

cd /tmp
wget http://ftp.cixug.es/apache/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
tar -xvzf spark-2.4.5-bin-hadoop2.7.tgz
wget https://downloads.lightbend.com/scala/2.12.6/scala-2.12.6.tgz
tar -xvzf scala-2.12.6.tgz

sudo mv /tmp/spark-2.4.5-bin-hadoop2.7/* /usr/local/spark
sudo mv /tmp/scala-2.12.6/* /usr/local/scala

echo 'export SCALA_HOME=/usr/local/scala' >> ~/.paths
echo 'export SPARK_HOME=/usr/local/spark' >> ~/.paths
echo 'export PATH=$SPARK_HOME/bin:$JAVA_HOME/bin:$SCALA_HOME/bin:$PATH' >> ~/.paths
source ~/.paths

cd /usr/local/spark/conf

sudo cp spark-env.sh.template spark-env.sh
echo 'export SCALA_HOME=/usr/local/scala'                 | sudo tee -a spark-env.sh
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' | sudo tee -a spark-env.sh
echo 'export SPARK_WORKER_MEMORY=1g'                      | sudo tee -a spark-env.sh
echo 'export SPARK_WORKER_INSTANCES=2'                    | sudo tee -a spark-env.sh
echo 'export SPARK_MASTER_IP=127.0.0.1'                   | sudo tee -a spark-env.sh
echo 'export SPARK_MASTER_PORT=7077'                      | sudo tee -a spark-env.sh
echo 'export SPARK_WORKER_DIR=/app/spark/tmp'             | sudo tee -a spark-env.sh

sudo cp spark-defaults.conf.template spark-defaults.conf
echo 'spark.master                     spark://localhost:7077' | sudo tee -a spark-defaults.conf

sudo cp slaves.template slaves
#echo localhost | sudo tee -a slaves

cd /usr/local/spark/sbin
sudo ./start-all.sh
#sudo ./stop-all.sh

