echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
sudo apt-get update
sudo apt-get install -y sbt

[ -e /media/DATA/sbt ] && ln -s /media/DATA/sbt/.sbt ~
[ -e /media/DATA/sbt ] && ln -s /media/DATA/sbt/.ivy2 ~
