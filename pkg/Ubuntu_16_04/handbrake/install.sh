sudo add-apt-repository -y ppa:stebbins/handbrake-releases
sudo apt-get update -y
sudo apt-get install -y handbrake-gtk handbrake-cli

mkdir -p ~/config
cp ./HandBrake.conf ~/config
cp ./presets.json ~/config
