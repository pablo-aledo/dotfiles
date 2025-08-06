cd
git clone https://github.com/wineasio/wineasio
cd wineasio
git clone https://github.com/falkTX/rtaudio
sudo apt install libjack-jackd2-dev wine-staging-dev
make 64
sudo mv build64/wineasio.dll /usr/lib/x86_64-linux-gnu/wine/wineasio.dll
sudo mv build64/wineasio.dll.so /usr/lib/x86_64-linux-gnu/wine/wineasio.dll.so
wine64 regsvr32 /usr/lib/x86_64-linux-gnu/wine/wineasio.dll
