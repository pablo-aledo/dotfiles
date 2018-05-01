sudo apt-get install -y lua5.2
cd /tmp
git clone https://github.com/zevv/lsofgraph.git
cd lsofgraph
sudo lsof -n -F | ./lsofgraph | unflatten -l 1 -c 6 | dot -T jpg > /tmp/a.jpg

cd /tmp
git clone https://github.com/akme/lsofgraph-python
cd lsofgraph-python
sudo lsof -n -F | python lsofgraph.py | unflatten -l 1 -c 6 | dot -T jpg > /tmp/a.jpg

