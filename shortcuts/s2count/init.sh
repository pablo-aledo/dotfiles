[ $# -ge 1 ] && query=$2
[ $# -ge 1 ] || query="is: starred avxhm.se"

echo $query > /tmp/query
