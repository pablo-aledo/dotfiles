cd
git clone https://github.com/wineasio/wineasio
cd wineasio
git clone https://github.com/falkTX/rtaudio

sudo apt install libjack-jackd2-dev
#sudo apt install libjack-jackd2-dev wine-staging-dev

make 64
sudo mv build64/wineasio64.dll    /usr/local/lib/wine/x86_64-windows/wineasio64.dll
sudo mv build64/wineasio64.dll.so /usr/local/lib/wine/x86_64-windows/wineasio64.dll.so
wine regsvr32 /usr/local/lib/wine/x86_64-windows/wineasio64.dll.so
