sudo apt-get install -y ccache

sudo /usr/sbin/update-ccache-symlinks

echo 'export PATH="/usr/lib/ccache:$PATH"' >> ~/.paths
source ~/.paths

ccache -F 0
ccache -M 0

ccache -s
ccache -C -z
