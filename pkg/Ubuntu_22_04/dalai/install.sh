cd
git clone https://github.com/cocktailpeanut/dalai.git
cd dalai

docker compose build
docker compose run dalai npx dalai alpaca install 7B # or a different model
docker compose up -d
