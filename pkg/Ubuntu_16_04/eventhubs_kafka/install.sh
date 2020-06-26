cd; mkdir aeh; cd aeh
wget https://github.com/Azure/azure-event-hubs-for-kafka/archive/master.zip
unzip master.zip
cd azure-event-hubs-for-kafka-master/tutorials/akka/java/producer
pkg install maven default-jdk
vim src/main/resources/application.conf
mvn clean package
mvn -e exec:java -Dexec.mainClass="AkkaTestProducer"

cp ~/.dotfiles/pkg/Ubuntu_16_04/eventhubs_kafka/pom.xml pom.xml
mvn clean compile assembly:single
