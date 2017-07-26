cd /usr/share
sudo mkdir WMail
cd WMail
sudo wget https://github.com/Thomas101/wmail/releases/download/v1.3.1/WMail_1_3_1_linux64.zip
sudo unzip WMail_1_3_1_linux64.zip
cd WMail-linux-x64
sudo ln -s $PWD/WMail /bin/wmail
[ -e /media/DATA/WMail ] && ln -s /media/DATA/WMail ~/.config/wmail
