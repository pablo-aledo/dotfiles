[ $# -ge 1 ] && tag=$1
[ $# -ge 1 ] || tag=select

echo $tag > /tmp/tag
