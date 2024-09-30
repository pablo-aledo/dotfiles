docker-compose up -d
./docker-make.sh
google-chrome http://localhost:5550
docker exec -it cats-gdbfrontend tmux a -t gdb-frontend
