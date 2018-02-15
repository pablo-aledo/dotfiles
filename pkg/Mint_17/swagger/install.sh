sudo apt-get -y install npm
sudo ln -s /usr/bin/nodejs /usr/bin/node

sudo npm install -g swagger

cd /usr/share/
sudo mkdir swagger-codegen
cd swagger-codegen
sudo wget https://oss.sonatype.org/content/repositories/releases/io/swagger/swagger-codegen-cli/2.2.1/swagger-codegen-cli-2.2.1.jar

echo 'swagger-codegen(){' >> ~/.paths
echo '	java -jar /usr/share/swagger-codegen/swagger-codegen-cli-2.2.1.jar $*' >> ~/.paths
echo '}' >> ~/.paths
source ~/.paths
