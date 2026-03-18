[ $# -eq 0 ] && [ -e /tmp/scount ] && count=$(cat /tmp/scount)
[ $# -eq 0 ] && [ -e /tmp/query  ] && query=$(cat /tmp/query)
[ $# -eq 0 ] && [ ! -e /tmp/scount ] && count=100
[ $# -eq 0 ] && [ ! -e /tmp/s2query ] && query="is: starred news.ycombinator.com"
[ $# -eq 0 ] && skip=0


[ $# -eq 1 ] && count=$1
[ $# -eq 1 ] && query="is: starred news.ycombinator.com"
[ $# -eq 1 ] &&  skip=0

[ $# -eq 2 ] && count=$1
[ $# -eq 2 ] && query=$2
[ $# -eq 2 ] &&  skip=0

[ $# -eq 3 ] && count=$1
[ $# -eq 3 ] && query=$2
[ $# -eq 3 ] &&  skip=$3

echo $query > /tmp/query
echo $count > /tmp/count
echo $skip > /tmp/skip

# script.google.com
# function subjectsPorBusqueda() {
#   const n = 10;
#   const query = 'query';
#
#   const threads = GmailApp.search(query, 0, n);
#
#   threads.forEach(t => {
#     const msg = t.getMessages()[0];
#     console.log(msg.getSubject());
#   });
# }
