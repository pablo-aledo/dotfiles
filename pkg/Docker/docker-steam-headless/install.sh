# Create a directory for your service:
sudo mkdir -p /opt/container-services/steam-headless
sudo chown -R $(id -u):$(id -g) /opt/container-services/steam-headless

# Create a directory for your service config data:
sudo mkdir -p /opt/container-data/steam-headless/{home,.X11-unix,pulse}
sudo chown -R $(id -u):$(id -g) /opt/container-data/steam-headless

#(Optional) Create a directory for your game install location:
sudo mkdir /mnt/games
sudo chmod -R 777 /mnt/games
sudo chown -R $(id -u):$(id -g) /mnt/games

# Create a Steam Headless /opt/container-services/steam-headless/docker-compose.yml file.
# Populate this file with the contents of the default Docker Compose File.
cp ./docker-compose.default.yml > /opt/container-services/steam-headless/docker-compose.yml

#Create a Steam Headless /opt/container-services/steam-headless/.env file with the contents found in this example Environment File.
cp ./env /opt/container-services/steam-headless/.env

cd /opt/container-services/steam-headless
sudo docker-compose up -d --force-recreate

echo http://$(myip | grep external | cut -d: -f2):8083/
