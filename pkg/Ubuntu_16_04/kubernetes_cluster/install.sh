# docker
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main'
sudo apt-get update
apt-cache policy docker-engine
sudo apt-get install -y docker-engine
sudo usermod -aG docker $(whoami)

# kubectl
sudo apt-get update && sudo apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo 'deb http://apt.kubernetes.io/ kubernetes-xenial main' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl

# minikube
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/

# helm
cd /tmp
wget https://kubernetes-helm.storage.googleapis.com/helm-v2.8.2-linux-amd64.tar.gz
tar -xvzf helm-v2.8.2-linux-amd64.tar.gz
sudo cp linux-amd64/helm /usr/bin

# start
sudo systemctl start docker
sudo /usr/local/bin/minikube start --vm-driver=none
sudo /usr/bin/helm init

# config
echo 'source <(kubectl completion zsh)' >> ~/.shell

#kubectl create secret docker-registry my-reg --docker-server=registry.gitlab.com --docker-username=<your-name> --docker-password=<your-pword> --docker-email=<your-email>
#kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "my-reg"}]}'
