sudo apt-get install -y curl openssh-server ca-certificates postfix
curl https://packages.gitlab.com/install/repositories/gitlab/gitlab-ce/script.deb.sh | sudo bash
sudo apt-get install -y gitlab-ce
sudo gitlab-ctl reconfigure

#IP=`wget http://ipinfo.io/ip -qO -`
IP=`sudo cat /etc/gitlab/gitlab.rb | egrep '^external_url' | sed -e 's/.*\/\///g' -e 's/.$//g'`

sudo sed -i "s/# gitlab_rails\['gitlab_email_from'\].*/gitlab_rails['gitlab_email_from'] = '$(whoami)@$IP'/g" /etc/gitlab/gitlab.rb

sudo gitlab-ctl reconfigure
sudo gitlab-ctl restart

sudo postsuper -d ALL

sudo apt-get install -y redis-server

echo '5iveL!fe' | xclip -sel clip
