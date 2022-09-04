curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

echo 'autoload -U +X bashcompinit && bashcompinit' >> ~/.paths
echo 'source ~/.dotfiles/pkg/Ubuntu_22_04/az/az.completion' >> ~/.paths
