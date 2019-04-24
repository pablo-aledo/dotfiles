cd src

rm -fr vet.report
find -name '*.go' | grep -v '/vendor/' | xargs dirname | sort | uniq > /tmp/folders

while read line
do
    go vet "$line" | tee -a vet.report
done < /tmp/folders

