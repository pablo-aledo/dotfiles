# sed -i 's/--frozen-lockfile/--no-frozen-lockfile/g' Dockerfile
docker compose build
docker compose up
# Go to http://localhost:8888
