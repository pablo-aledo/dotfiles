#sudo apt-get install -y awscli
sudo snap install aws-cli --classic

echo 'autoload bashcompinit && bashcompinit' >> ~/.paths
echo 'autoload -Uz compinit && compinit' >> ~/.paths
echo 'complete -C /usr/bin/aws_completer aws' >> ~/.paths
source ~/.paths
