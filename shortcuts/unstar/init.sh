[ $# -ge 1 ] && tag=$1
[ $# -ge 1 ] || tag=star

echo $tag > /tmp/tag
