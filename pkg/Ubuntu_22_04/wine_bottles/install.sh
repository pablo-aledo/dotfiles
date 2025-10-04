sudo apt update
sudo apt install flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install flathub com.usebottles.bottles

cd ~/.var/app/com.usebottles.bottles/data/bottles/runners
wget https://github.com/GloriousEggroll/proton-ge-custom/releases/download/GE-Proton10-17/GE-Proton10-17.tar.gz -O - | tar -xvz

flatpak run com.usebottles.bottles
