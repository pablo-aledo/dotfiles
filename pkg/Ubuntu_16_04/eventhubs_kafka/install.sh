cd; mkdir aeh; cd aeh
wget https://github.com/Azure/azure-event-hubs-for-kafka/archive/master.zip
unzip master.zip
cd azure-event-hubs-for-kafka-master/quickstart/java/producer
pkg install maven default-jdk
vim src/main/resources/producer.config
mvn clean package
mvn -e exec:java -Dexec.mainClass="TestProducer"

cp ~/.dotfiles/pkg/Ubuntu_16_04/eventhubs_kafka/pom.xml pom.xml
mvn clean compile assembly:single
