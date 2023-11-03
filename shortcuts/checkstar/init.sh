[ $# -ge 1 ] && filter=$1 || filter='label: select'
[ $# -ge 2 ] && file=$2 || file=/tmp/titles

echo $filter > /tmp/checkstar_filter
[ $file != /tmp/titles ] && \cp $file /tmp/titles

Rm () {
        [ "$1" = "-rf" ] && dir=~/.Trash/`date +%s`  && mkdir -p $dir && shift && /bin/mv $* $dir && return
        [ "$1" = "-f" ] && dir=~/.Trash/`date +%s`  && mkdir -p $dir && shift && /bin/mv $* $dir && return
        /bin/rm --one-file-system -i $*
}

Rm -rf /tmp/checkstar
