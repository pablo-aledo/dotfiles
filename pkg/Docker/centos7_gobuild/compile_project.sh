cd $GOPATH/src/*/
dep ensure

find $GOPATH/src -name cmd -type d | while read line
do
    cd $line
    go build -o ~/go/bin/$(basename $line) -i $(echo $line | sed "s|$GOPATH/src/||g")
done
