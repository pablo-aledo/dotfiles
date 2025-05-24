mkdir ~/.vst ~/.vst3 ~/.vstwin
( cd ~/.local; wget https://github.com/robbert-vdh/yabridge/releases/download/5.0.5/yabridge-5.0.5.tar.gz -O - | tar -xz )
cd ~/.local/yabridge
./yabridgectl set --path $PWD
./yabridgectl add ~/.vstwin
./yabridgectl sync

# Native Access 1.14.1
# sudo mount -t udf '<plugin>.iso' -o unhide /mnt/cdrom0
