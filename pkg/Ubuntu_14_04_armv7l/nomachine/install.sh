source ~/.dotfiles/source/.shell/links

cd /tmp/

for a in `seq 1 15`; do wget "https://www.nomachine.com/download/download&id=$a" ; done
dl=$(for a in `seq 1 15`; do links download\&id=$a ; done | grep armhf.deb | head -n1)

wget $dl

sudo dpkg -i `basename $dl`
