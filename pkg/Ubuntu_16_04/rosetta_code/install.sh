mkdir tasks
cd tasks
wget www.rosettacode.org/wiki/Category:Programming_Tasks -O tasks
wget -c $(links tasks | grep wiki | grep -v : | sort | uniq | grep -v mediawiki | sed 's/^/www.rosettacode.org/g')
cd ..

mkdir languages
cd languages
wget www.rosettacode.org/wiki/Category:Programming_Languages -O languages
wget -c $(links languages | grep wiki | sort | uniq | grep "/wiki/Category:" | sed 's/^/www.rosettacode.org/g')
cd ..

mkdir html
cd tasks
ls | grep -v tasks | while read line
do
    cp "$line" "../html/$line.html"
done
cd ..

mkdir text
cd html
ls | while read line
do
    w3m -dump "$line" > "../text/$(basename $line .html)"
done
cd ..

mkdir text_classified
cd text
ls | while read line
do
    mkdir "../text_classified/$line"
done
cd ../text_classified
ls | while read line
do
    cd $line
    csplit -s "../../text/$line" '/^.*\[edit\]$/' '{*}'
    cd ..
done
cd ..

cd text_classified
ls | while read line
do
    cd "$line"
    ls | while read src
    do
        name=$(head -n1 $src | sed -e 's/.edit.//g' -e 's|/|.|g')
        n=0; dst=$name
        while [ -e $dst ]
        do
            dst=$name.$n
            n=$(( $n + 1 ))
        done
        mv "$src" "$dst"
    done
    cd ..
done
cd ..

