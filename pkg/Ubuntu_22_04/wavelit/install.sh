sudo wget https://github.com/seqeralabs/wavelit/releases/download/v0.7.0/wavelit-0.7.0-linux-x86_64 -o /usr/bin/wavelit
sudo chmod +x /usr/bin/wavelit

Wavelit(){
    docker run --privileged -w $2 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -it $(wavelit --config-file https://fusionfs.seqera.io/releases/v2.2.7-amd64.json --conda-package="conda-forge::procps-ng $1")
}
