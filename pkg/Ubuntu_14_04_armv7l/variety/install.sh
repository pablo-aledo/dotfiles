sudo apt-get install -y software-properties-common python-software-properties
sudo add-apt-repository ppa:peterlevi/ppa
sudo apt-get update
sudo apt-get install -y variety


mkdir -p ~/Variety 
ln -s ~/Variety ~/.config/variety
cp .firstrun ~/.config/variety/
cp smart_user.json ~/.config/variety/
ln -s $PWD/variety.conf ~/.config/variety/

echo 'WP=$1'                            >  ~/.config/variety/scripts/set_wallpaper
echo 'feh --bg-fill "$WP" 2> /dev/null' >> ~/.config/variety/scripts/set_wallpaper

