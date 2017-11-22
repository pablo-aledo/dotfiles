#source ~/.dotfiles/source/.shell/links

cd /tmp/

for a in `seq 1 15`; do wget "https://www.nomachine.com/download/download&id=$a" ; done

dl=`cat download\&id=* | grep 'amd64' | grep /download/ | grep -v Client | grep -v client | head -n1 | cut -d"'" -f2`

wget $dl

sudo dpkg -i `basename $dl`
