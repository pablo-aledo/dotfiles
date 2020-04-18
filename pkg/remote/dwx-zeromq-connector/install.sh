mkd /tmp/dwx-zeromq-connector
wget https://github.com/darwinex/dwx-zeromq-connector/archive/master.zip
unzip master.zip
unzip dwx-zeromq-connector-master/dependencies/mql-zmq-master.zip
mkd c
appid=1DAFD9A7C67DC84FE37EAA1FC1E5CF75

mkdir -p Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Experts
cp -r ../dwx-zeromq-connector-master/v2.0.1/mql4/DWX_ZeroMQ_Server_v2.0.1_RC8.mq4 Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Experts

mkdir -p Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Include
cp -r ../mql-zmq-master/Include/* Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Include

mkdir -p Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Libraries
cp -r ../mql-zmq-master/Library/MT4/* Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Libraries

mkdir -p Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Scripts
cp -r ../mql-zmq-master/Scripts/* Users/$USER/AppData/Roaming/MetaQuotes/Terminal/$appid/MQL4/Scripts

zip -r mql-zmq-master.zip *

touch .htaccess
sudo docker run -d --rm -p 80:80 -v $PWD:/var/www/html fauria/lamp
echo "Download and extract to C:"; read
sudo docker ps | grep fauria.lamp | awk '{print $1}' | xargs sudo docker kill
