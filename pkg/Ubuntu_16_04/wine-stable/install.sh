sudo add-apt-repository ppa:ricotz/unstable
pkg update
pkg install wine-stable

sudo dpkg --add-architecture i386 && sudo apt-get update && sudo apt-get install wine32

sudo apt-get install winbind
