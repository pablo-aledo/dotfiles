cd
mkdir puter
cd puter
mkdir -p puter/config puter/data
sudo chown -R 1000:1000 puter
docker run --rm -p 4100:4100 -v `pwd`/puter/config:/etc/puter -v `pwd`/puter/data:/var/puter  ghcr.io/heyputer/puter
