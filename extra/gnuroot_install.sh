apt-get update
apt-get upgrade

apt-get install -y lxde-core
apt-get install -y xterm synaptic pulseaudio
apt-get install -y wget

cat << EOF > /bin/starti3
export DISPLAY=:0 PULSE_SERVER=tcp:127.0.0.1:4712
i3
EOF
chmod +x /bin/starti3

cd ~
wget https://raw.githubusercontent.com/pablo-aledo/dotfiles/master/source/.shell/pkg -O .pkg

echo 'pkg update'         >> .pkg
echo 'pkg install unzip'  >> .pkg
echo 'pkg install zsh'    >> .pkg
echo 'pkg install git'    >> .pkg
echo 'pkg install i3'    >> .pkg
echo 'pkg install feh'    >> .pkg
echo 'wget https://github.com/pablo-aledo/dotfiles/archive/master.zip -O .dotfiles.zip' >> .pkg
echo 'unzip .dotfiles.zip' >> .pkg
echo 'mv dotfiles-master .dotfiles' >> .pkg

echo 'ROOT=/home/.dotfiles' >> .pkg
echo 'HOME=/home' >> .pkg
ROOT=/home/.dotfiles
HOME=/home

head $ROOT/install.sh -n60 | tail -n55 >> .pkg

echo 'link(){' >> .pkg
echo '	echo -e "\e[34m linking file ... \e[0m" `basename $1`' >> .pkg
echo '	src=$1' >> .pkg
echo '	dst=$(echo $1 | sed "s/`escape $ROOT`\/link/`escape $HOME`/g")' >> .pkg
echo '	mkdir -p $(dirname $dst)' >> .pkg
echo '	cp $src $dst' >> .pkg
echo '}' >> .pkg

tail $ROOT/install.sh -n35 >> .pkg

bash .pkg



