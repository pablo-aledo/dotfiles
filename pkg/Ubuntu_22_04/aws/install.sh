#sudo apt-get install -y awscli
#sudo snap install aws-cli --classic

cd
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

echo 'autoload bashcompinit && bashcompinit' >> ~/.paths
echo 'autoload -Uz compinit && compinit' >> ~/.paths
echo 'complete -C /usr/local/bin/aws_completer aws' >> ~/.paths
source ~/.paths
