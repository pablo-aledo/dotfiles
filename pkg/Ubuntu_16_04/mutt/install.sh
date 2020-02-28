export DEBIAN_FRONTEND=noninteractive
echo "postfix postfix/mailname string example.com" | sudo debconf-set-selections
echo "postfix postfix/main_mailer_type string 'Internet Site'" | sudo debconf-set-selections
sudo apt-get -yq install mutt
