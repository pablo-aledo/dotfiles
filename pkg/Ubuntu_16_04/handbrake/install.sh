sudo add-apt-repository ppa:stebbins/handbrake-releases
sudo apt-get update
sudo apt-get install handbrake-gtk handbrake-cli

mkdir -p ~/config
cp ./HandBrake.conf ~/config
cp ./presets.json ~/config
