cd
git clone https://github.com/ahmedkhaleel2004/gitdiagram.git
cd gitdiagram

pnpm i
cp .env.example .env
#docker compose up --build -d
docker compose up --build

chmod +x start-database.sh
./start-database.sh
#yes
pnpm db:push

pnpm dev
