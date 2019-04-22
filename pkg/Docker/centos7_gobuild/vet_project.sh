cd src

find -name '*.go' | grep -v '/vendor/' | xargs dirname | sort | uniq | while read line
do
    go vet "$line"
done

