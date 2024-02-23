[ $# -ge 1 ] && filter=$1 || filter='label: select'
[ $# -ge 2 ] && file=$2 || file=/tmp/titles

echo $filter > /tmp/checkstar_filter
[ $file != /tmp/titles ] && \cp $file /tmp/titles

xdotool key 6
