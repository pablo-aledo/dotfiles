sudo apt update
sudo apt install jackd2 qjackctl pulseaudio-module-jack
sudo usermod -aG audio $USER

sudo apt install hydrogen

echo '/usr/bin/jackd -dalsa' > ~/.jackdrc
pactl load-module module-jack-sink
pactl load-module module-jack-source
