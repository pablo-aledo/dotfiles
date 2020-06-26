cd; mkdir aeh; cd aeh
wget https://github.com/Azure/azure-event-hubs-for-kafka/archive/master.zip
unzip master.zip
cd azure-event-hubs-for-kafka-master/tutorials/akka/java/producer
pkg install maven default-jdk
vim src/main/resources/application.conf
mvn clean package
mvn -e exec:java -Dexec.mainClass="AkkaTestProducer"
