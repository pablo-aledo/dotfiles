[ $# -ge 1 ] && query=$1
[ $# -ge 1 ] || query="is: starred xsava.xyz"

echo $query > /tmp/query
