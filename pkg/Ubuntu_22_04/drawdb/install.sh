cd
git clone https://github.com/drawdb-io/drawdb.git
cd drawdb
docker build -t drawdb .
docker run -p 3000:80 drawdb
