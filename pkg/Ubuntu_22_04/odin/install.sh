cd ~/obsidian/.obsidian/plugins
git clone https://github.com/memgraph/odin.git
cd odin
echo "OPENAI_API_KEY=SOPENAI_API_KEY" > .env
docker compose up
