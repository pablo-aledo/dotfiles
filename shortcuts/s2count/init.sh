[ $# -ge 1 ] && query=$1
[ $# -ge 1 ] || query="is: starred avxhm.se"

echo $query > /tmp/query
