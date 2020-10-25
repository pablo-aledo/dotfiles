mkdir ~/midi2ascii
cd ~/midi2ascii
wget http://www.archduke.org/midi/asc2mid.c
wget http://www.archduke.org/midi/mid2asc.c
wget http://www.archduke.org/midi/instrux.html
gcc asc2mid.c -o asc2mid
gcc mid2asc.c -lm -o mid2asc
