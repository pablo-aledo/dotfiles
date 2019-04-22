cd src

find -name '*.go' | xargs dirname | sort | uniq | while read line
do
    go vet "$line"
done

