sudo apt-get install -y awscli

echo 'autoload bashcompinit && bashcompinit' >> ~/.paths
echo 'autoload -Uz compinit && compinit' >> ~/.paths
echo 'complete -C /usr/bin/aws_completer aws' >> ~/.paths
source ~/.paths
