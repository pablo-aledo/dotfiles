cd src

rm -fr vet.report
find -name '*.go' | grep -v '/vendor/' | xargs dirname | sort | uniq > /tmp/folders

while read line
do
    go vet "$line" 2>&1 | tee -a vet.report
done < /tmp/folders

