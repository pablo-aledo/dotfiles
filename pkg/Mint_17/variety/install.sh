sudo add-apt-repository ppa:peterlevi/ppa -y
sudo apt-get update
sudo apt-get install -y variety

if [ -e /media/DATA/Variety ]
then
	ln -s /media/DATA/Variety ~/.config/variety
else
	mkdir -p /media/DATA/Variety 
	ln -s /media/DATA/Variety ~/.config/variety
	cp .firstrun ~/.config/variety/
	cp smart_user.json ~/.config/variety/
	ln -s $PWD/variety.conf ~/.config/variety/
fi
