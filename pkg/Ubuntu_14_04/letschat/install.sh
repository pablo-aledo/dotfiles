sudo apt-get install -y nodejs npm nodejs-legacy mongodb mongodb-server git
cd ~
git clone https://github.com/sdelements/lets-chat.git
cd lets-chat
npm install
cp settings.yml.sample settings.yml
