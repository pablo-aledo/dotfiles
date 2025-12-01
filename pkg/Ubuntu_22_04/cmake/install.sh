sudo apt update
sudo apt install -y software-properties-common lsb-release apt-transport-https ca-certificates gnupg

sudo apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo add-apt-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

sudo apt update
sudo apt install -y cmake
