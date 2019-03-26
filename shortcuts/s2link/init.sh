[ $# -ge 1 ] && count=$1
[ $# -ge 1 ] || count=100

[ $# -ge 2 ] && query=$2
[ $# -ge 2 ] || query="is: starred avxhm.se"

echo $query > /tmp/query
echo $count > /tmp/count
